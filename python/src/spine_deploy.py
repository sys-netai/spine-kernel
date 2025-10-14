from enum import Enum
import os
import stat
import sys
import json
import time
import argparse
import threading
from functools import partial
import context
import numpy as np
import json
import time
from collections import OrderedDict

from logger import logger as log
from message import *
from netlink import Netlink
from spine_flow import Flow, ActiveFlowMap, EnvFlows
from poller import Action, Poller, ReturnStatus, PollEvents
from helper import drop_privileges
import msg_sender

# import from main repo
from agent.policycache import DoubleTree, TreeType
from agent.definitions import transform_state, map_action, Direction, get_last_received_id, get_last_learned_id, get_last_learned_direction, update_score, ACTION_DIM, STATE_DIM

# log.setLevel("INFO")
# communication between spine-user-space and kernel
nl_sock = None
# kernel cc alg
kernel_cc = None
# message sender
nl_send = None
# cont status of polling
cont = threading.Event()
poller = Poller()

# we do require one active flow map here
env_flows = EnvFlows()
env_id = "spine_eval"
global_flow_id = 0

# Model inference related constants and variables
START_UPDATE_SCORE_STEP = 5
STATE_BUFFER_SIZE = 50
INITIAL_SCORE = 1
STACK_LENGTH = 3
DEFAULT_PROB = [0, 1]
PROBE_THRESHOLD = 0.65
SLOW_START_THRESHOLD = 0.65


# Use static and dynamic models
RUN_STATIC = False
RUN_DYNAMIC = True

# Global model and state management
double_tree = None
flow_states = {}  # per-flow state management
flow_state_buffers = {}  # per-flow state buffers: sock_id -> OrderedDict()
flow_step_counters = {}  # per-flow step counters: sock_id -> step_count
flow_score_states = {}  # per-flow score states: sock_id -> {'dt_score': x, 'vfdt_score': y}
flow_slow_starts = {}  # per-flow slow start flags: sock_id -> bool

# Batch processing variables
ready_for_inference = {}  # sock_id -> (neo_state, processed_obs, step_counter, timestamp)
batch_processing_interval = 0.005  # 5ms in seconds
last_batch_time = 0
batch_processing_lock = threading.Lock()

# Periodic MeasureMsg sending variables
active_sock_ids = set()  # Track active sock_ids for periodic MeasureMsg sending
measure_interval = 0.03  # 30ms in seconds
last_measure_time = 0
measure_lock = threading.Lock()
request_id_counter = 0

# Timeout mechanism variables
flow_last_response_time = {}  # sock_id -> last_response_timestamp
timeout_threshold = 5.0  # 5 seconds timeout in seconds
timeout_check_interval = 1.0  # Check for timeouts every 1 second
last_timeout_check = 0
timeout_lock = threading.Lock()


def build_netlink_sock():
    sock = Netlink()
    sock.add_mc_group()
    return sock


def initialize_double_tree():
    """Initialize the DoubleTree model for inference"""
    global double_tree
    double_tree = DoubleTree(
        dt_path="dt_tree.pkl",
        online_tree_type=TreeType.HAT,
        n_classes=[0, 1],
        using_dt=False,  # Disable decision tree for cost savings
    )
    log.info("DoubleTree model initialized successfully")


def get_or_create_flow_data(sock_id):
    """Get or create per-flow data structures"""
    global flow_states, flow_state_buffers, flow_step_counters, flow_score_states, flow_slow_starts
    
    if sock_id not in flow_states:
        flow_states[sock_id] = {}
        flow_state_buffers[sock_id] = OrderedDict()
        flow_step_counters[sock_id] = 0
        flow_score_states[sock_id] = {'dt_score': INITIAL_SCORE, 'vfdt_score': INITIAL_SCORE}
        flow_slow_starts[sock_id] = True
        log.debug("Created new flow data structures for sock_id: {}".format(sock_id))
    
    return (flow_states[sock_id], flow_state_buffers[sock_id], flow_step_counters[sock_id],
            flow_score_states[sock_id], flow_slow_starts[sock_id])


def prepare_single_flow_for_inference(neo_state, sock_id):
    """Prepare a single flow state for inference immediately"""
    global flow_states, flow_state_buffers, flow_step_counters, batch_processing_lock, ready_for_inference
    
    # Get or create per-flow data structures
    flow_states, state_buffer, step_counter, score_state, slow_start = get_or_create_flow_data(sock_id)
    
    step_counter += 1
    flow_step_counters[sock_id] = step_counter
    
    # Transform state using per-flow state
    obs, flow_states = transform_state(neo_state, flow_states)
    
    # Handle state history (simplified version)
    if 'state_history' not in flow_states:
        flow_states['state_history'] = []
    
    flow_states['state_history'].append(obs)
    if len(flow_states['state_history']) > 20:
        flow_states['state_history'].pop(0)
    
    # Optimized state concatenation
    if len(flow_states['state_history']) >= STACK_LENGTH:
        processed_obs = np.array(flow_states['state_history'][-STACK_LENGTH:]).flatten()
    else:
        processed_obs = np.zeros(STATE_DIM * STACK_LENGTH, dtype=np.float32)
    
    # Add to ready for inference queue
    with batch_processing_lock:
        ready_for_inference[sock_id] = (neo_state, processed_obs, step_counter, time.time())
    
    log.debug("Prepared flow {} for inference, step_counter: {}".format(sock_id, step_counter))


def collect_ready_inferences():
    """Collect all ready inferences for batch processing"""
    global ready_for_inference, batch_processing_lock
    
    with batch_processing_lock:
        if not ready_for_inference:
            return []
        
        # Copy ready inferences and clear the queue
        current_batch = ready_for_inference.copy()
        ready_for_inference.clear()
    
    return current_batch


def update_flow_response_time(sock_id):
    """Update the last response time for a flow"""
    global flow_last_response_time, timeout_lock
    
    with timeout_lock:
        flow_last_response_time[sock_id] = time.time()
        log.debug("Updated response time for sock_id: {}".format(sock_id))


def check_and_remove_timeout_flows():
    """Check for flows that have timed out and remove them"""
    global flow_last_response_time, active_sock_ids, timeout_threshold, timeout_check_interval, last_timeout_check, timeout_lock, measure_lock
    
    current_time = time.time()
    
    # Check if enough time has passed for timeout checking
    if current_time - last_timeout_check < timeout_check_interval:
        return
    
    with timeout_lock:
        timed_out_flows = []
        for sock_id, last_response_time in flow_last_response_time.items():
            if current_time - last_response_time > timeout_threshold:
                timed_out_flows.append(sock_id)
        
        # Remove timed out flows
        for sock_id in timed_out_flows:
            log.warning("Flow sock_id {} timed out (no response for {:.2f}s), removing from active flows".format(
                sock_id, current_time - flow_last_response_time[sock_id]))
            
            # Remove from response time tracking
            del flow_last_response_time[sock_id]
            
            # Remove from active flows
            with measure_lock:
                active_sock_ids.discard(sock_id)
            
            # Clean up per-flow data structures
            cleanup_flow_data(sock_id)
            
            # Release from env_flows
            env_flows.release_sock_id_to_env(sock_id)
    
    last_timeout_check = current_time


def send_periodic_measure_requests():
    """Send periodic MeasureMsg requests every 30ms for active flows"""
    global active_sock_ids, measure_interval, last_measure_time, measure_lock, request_id_counter, nl_send, nl_sock
    
    current_time = time.time()
    
    # Check if enough time has passed for sending MeasureMsg requests
    if current_time - last_measure_time < measure_interval:
        return
    
    with measure_lock:
        if not active_sock_ids:
            return
        
        # Send MeasureMsg request for each active sock_id
        for sock_id in active_sock_ids:
            try:
                request_id_counter += 1
                nl_send(request_id_counter, nl_sock, sock_id, msg_type=NL_MEASURE)
                log.debug("Sent periodic MeasureMsg request for sock_id: {}, request_id: {}".format(sock_id, request_id_counter))
            except Exception as e:
                log.error("Error sending periodic MeasureMsg for sock_id {}: {}".format(sock_id, e))
    
    last_measure_time = current_time




def batch_model_inference(ready_inferences):
    """Perform batch model inference for all ready flows"""
    global double_tree
    
    if double_tree is None:
        log.error("DoubleTree model not initialized")
        return []
    
    if not ready_inferences or len(ready_inferences) == 0:
        return []
    
    # Extract all observations for batch inference
    observations = np.array([data[1] for data in ready_inferences.values()])  # processed_obs is at index 1
    
    # Perform batch inference
    dt_action_probs, vfdt_action_probs = double_tree.predict_prob(observations)
    
    results = []
    for i, (sock_id, (neo_state, processed_obs, step_counter, timestamp)) in enumerate(ready_inferences.items()):
        # Handle DT probabilities
        if dt_action_probs is None or i >= len(dt_action_probs):
            dt_recommend_action_prob = DEFAULT_PROB
        else:
            dt_prob_sum = np.sum(dt_action_probs[i])
            if len(dt_action_probs[i]) < 2 or dt_prob_sum < 0.99:
                dt_recommend_action_prob = DEFAULT_PROB
            else:
                dt_recommend_action_prob = dt_action_probs[i].tolist()
        
        # Handle VFDT probabilities
        if vfdt_action_probs is None or i >= len(vfdt_action_probs):
            vfdt_recommend_action_prob = DEFAULT_PROB
        else:
            vfdt_prob_sum = np.sum(vfdt_action_probs[i])
            if len(vfdt_action_probs[i]) < 2 or vfdt_prob_sum < 0.99:
                vfdt_recommend_action_prob = DEFAULT_PROB
            else:
                vfdt_recommend_action_prob = vfdt_action_probs[i].tolist()
        
        # Calculate action values
        dt_action = - dt_recommend_action_prob[0] + dt_recommend_action_prob[1] + 1e-4
        vfdt_action = - vfdt_recommend_action_prob[0] + vfdt_recommend_action_prob[1] + 1e-4
        
        results.append({
            'sock_id': sock_id,
            'neo_state': neo_state,
            'processed_obs': processed_obs,
            'step_counter': step_counter,
            'dt_action_prob': dt_recommend_action_prob,
            'vfdt_action_prob': vfdt_recommend_action_prob,
            'dt_action': dt_action,
            'vfdt_action': vfdt_action
        })
    
    return results


def select_action_for_flow(inference_result):
    """Select action for a flow based on inference results (fast path for action sending)"""
    global flow_states, flow_score_states, flow_slow_starts
    
    sock_id = inference_result['sock_id']
    neo_state = inference_result['neo_state']
    dt_action = inference_result['dt_action']
    vfdt_action = inference_result['vfdt_action']
    
    # Get per-flow data structures
    flow_states, _, _, score_state, slow_start = get_or_create_flow_data(sock_id)
    
    # Action selection logic (fast path)
    dt_score = score_state['dt_score']
    vfdt_score = score_state['vfdt_score']
    
    
    if not RUN_STATIC:
        dt_score = 0
    if not RUN_DYNAMIC:
        vfdt_score = 0
    if dt_score == 0 and vfdt_score == 0:
        action = Direction.PROBE
    
    # Check last action for probe continuation
    if 'last_action' not in flow_states:
        flow_states['last_action'] = Direction.IDK
    
    last_learned_direction = get_last_learned_direction(neo_state)
    if flow_states['last_action'] == Direction.PROBE and last_learned_direction == Direction.PROBE:
        action = Direction.PROBE
    else:
        max_score = max(dt_score, vfdt_score)
        if max_score < PROBE_THRESHOLD:
            action = Direction.PROBE
        else:
            action = dt_action if dt_score > vfdt_score else vfdt_action
    
    # Ensure action is never None
    action = action if action is not None else Direction.IDK
    flow_states['last_action'] = action
    
    return action


def update_learning_for_flow(inference_result):
    """Update learning, scores, and model for a flow (slower path for learning updates)"""
    global flow_states, flow_state_buffers, flow_score_states, flow_slow_starts, double_tree
    
    sock_id = inference_result['sock_id']
    neo_state = inference_result['neo_state']
    processed_obs = inference_result['processed_obs']
    step_counter = inference_result['step_counter']
    dt_recommend_action_prob = inference_result['dt_action_prob']
    vfdt_recommend_action_prob = inference_result['vfdt_action_prob']
    
    # Get per-flow data structures
    flow_states, state_buffer, _, score_state, slow_start = get_or_create_flow_data(sock_id)
    
    # Update per-flow state buffer
    last_received_id = get_last_received_id(neo_state)
    last_learned_id = get_last_learned_id(neo_state)
    last_learned_direction = get_last_learned_direction(neo_state)
    
    state_buffer[last_received_id] = {
        'step': step_counter, 
        'state': processed_obs,
        'dt_prob': dt_recommend_action_prob,
        'vfdt_prob': vfdt_recommend_action_prob,
        'sock_id': sock_id
    }
    if len(state_buffer) > STATE_BUFFER_SIZE:
        state_buffer.popitem(last=False)
    
    # Check for learning update using per-flow buffer
    last_learned = None
    if last_learned_id in state_buffer:
        last_learned = state_buffer[last_learned_id]
        state_buffer.pop(last_learned_id)
    
    # Update model and scores
    if step_counter > START_UPDATE_SCORE_STEP:
        try:
            if last_learned is not None and (last_learned_direction == Direction.UP or last_learned_direction == Direction.DOWN):
                double_tree.update_ot(last_learned['state'], last_learned_direction)
                update_score(
                    score_state, last_learned_direction, last_learned['dt_prob'], last_learned['vfdt_prob']
                )
                if (score_state['vfdt_score'] < SLOW_START_THRESHOLD) and slow_start:
                    slow_start = False
                    log.info("SLOW START is over at step: {} for flow: {}".format(step_counter, sock_id))
        except Exception as e:
            log.error("Error in updating score for flow {}: {}".format(sock_id, e))


def process_batch_inferences():
    """Process all ready inferences in batch with prioritized action sending"""
    global batch_processing_lock, last_batch_time, flow_states, flow_slow_starts
    
    current_time = time.time()
    
    # Check if enough time has passed for batch processing
    if current_time - last_batch_time < batch_processing_interval:
        return
    
    # Collect all ready inferences
    ready_inferences = collect_ready_inferences()
    
    if not ready_inferences:
        return
    
    last_batch_time = current_time
    
    log.debug("Processing batch of {} inferences".format(len(ready_inferences)))
    
    try:
        # Step 1: Perform batch model inference
        inference_results = batch_model_inference(ready_inferences)
        
        # Step 2: Fast path - Send actions to kernel immediately
        for inference_result in inference_results:
            try:
                sock_id = inference_result['sock_id']
                neo_state = inference_result['neo_state']
                
                # Select action quickly (no learning updates)
                action = select_action_for_flow(inference_result)
                
                # Get per-flow states for action mapping
                if sock_id in flow_states:
                    per_flow_states = flow_states[sock_id]
                else:
                    per_flow_states = {}
                
                # Get per-flow slow_start state
                if sock_id in flow_slow_starts:
                    per_flow_slow_start = flow_slow_starts[sock_id]
                else:
                    per_flow_slow_start = True
                
                # Map action to network action format and send back to kernel immediately
                a, _ = map_action(action, neo_state, per_flow_states, slow_start=per_flow_slow_start)
                nl_send(a, nl_sock, sock_id)
                
                log.debug("Action sent for sock_id: {}, action: {}".format(sock_id, a))
                
            except Exception as e:
                log.error("Error in action selection/sending for sock_id {}: {}".format(sock_id, e))
                # Send default action on error
                try:
                    default_action = {"neo_action": 1000}
                    nl_send(default_action, nl_sock, sock_id)
                except Exception as send_error:
                    log.error("Error sending default action for sock_id {}: {}".format(sock_id, send_error))
        
        # Step 3: Slow path - Update learning, scores, and model (after actions are sent)
        for inference_result in inference_results:
            try:
                update_learning_for_flow(inference_result)
                log.debug("Learning updated for sock_id: {}".format(inference_result['sock_id']))
            except Exception as e:
                log.error("Error in learning update for sock_id {}: {}".format(inference_result['sock_id'], e))
    
    except Exception as e:
        log.error("Error in batch inference processing: {}".format(e))
        # Send default actions for all flows in case of batch processing error
        for sock_id, (neo_state, processed_obs, step_counter, timestamp) in ready_inferences.items():
            try:
                default_action = {"neo_action": 1000}
                nl_send(default_action, nl_sock, sock_id)
            except Exception as send_error:
                log.error("Error sending default action for sock_id {}: {}".format(sock_id, send_error))


def read_netlink_message(nl_sock: Netlink):
    hdr_raw = nl_sock.next_msg()
    if hdr_raw == None:
        return ReturnStatus.Cancel
    hdr = SpineMsgHeader()
    if hdr.from_raw(hdr_raw) == None:
        log.error("Failed to parse netlink header")
        return ReturnStatus.Cancel
    if hdr.type == NL_CREATE:
        print("NL_CREATE")
        msg = CreateMsg()
        msg.from_raw(hdr_raw[hdr.hdr_len :])
        flow = Flow().from_create_msg(msg, hdr)
        # register new flow
        active_flow_map = env_flows.get_env_flows(env_id)
        if active_flow_map == None:
            log.warn("env: {} has not registered".format(env_id))
            return ReturnStatus.Continue
        active_flow_map.add_flow_with_sockId(flow)
        # cache sockID with envid
        env_flows.bind_sock_id_to_env(flow.sock_id, env_id)
        
        # Add sock_id to active set for periodic MeasureMsg sending
        with measure_lock:
            active_sock_ids.add(flow.sock_id)
            log.debug("Added sock_id {} to active flows for periodic MeasureMsg sending".format(flow.sock_id))
        
        # Initialize response time tracking for the new flow
        update_flow_response_time(flow.sock_id)
        
        return ReturnStatus.Continue
    elif hdr.type == NL_READY:
        log.info("Spine kernel is ready!!")
    elif hdr.type == NL_MEASURE:
        sock_id = hdr.sock_id
        
        # Update response time for this flow
        update_flow_response_time(sock_id)
        
        # Parse the measure message
        msg = MeasureMsg()
        msg.from_raw(hdr_raw[hdr.hdr_len :])
        neo_state = msg.data
        
        # Immediately prepare flow state for inference
        prepare_single_flow_for_inference(neo_state, sock_id)
        
        log.debug("Prepared flow {} for immediate batch inference".format(sock_id))
        
    elif hdr.type == NL_RELEASE:        
        # flow release
        sock_id = hdr.sock_id
        # we just remove the cached items
        env_flows.release_sock_id_to_env(sock_id)
        
        # Remove sock_id from active set for periodic MeasureMsg sending
        with measure_lock:
            active_sock_ids.discard(sock_id)
            log.debug("Removed sock_id {} from active flows for periodic MeasureMsg sending".format(sock_id))
        
        # Remove from response time tracking
        with timeout_lock:
            flow_last_response_time.pop(sock_id, None)
            log.debug("Removed sock_id {} from response time tracking".format(sock_id))
        
        # Clean up per-flow data structures
        cleanup_flow_data(sock_id)
        
        # env has been deregistered, do nothing
    return ReturnStatus.Continue


def cleanup_flow_data(sock_id):
    """Clean up per-flow data structures when flow is released"""
    global flow_states, flow_state_buffers, flow_step_counters, flow_score_states, flow_slow_starts, flow_last_response_time
    
    if sock_id in flow_states:
        del flow_states[sock_id]
        log.debug("Cleaned up flow_states for sock_id: {}".format(sock_id))
    
    if sock_id in flow_state_buffers:
        del flow_state_buffers[sock_id]
        log.debug("Cleaned up flow_state_buffers for sock_id: {}".format(sock_id))
    
    if sock_id in flow_step_counters:
        del flow_step_counters[sock_id]
        log.debug("Cleaned up flow_step_counters for sock_id: {}".format(sock_id))
    
    if sock_id in flow_score_states:
        del flow_score_states[sock_id]
        log.debug("Cleaned up flow_score_states for sock_id: {}".format(sock_id))
    
    if sock_id in flow_slow_starts:
        del flow_slow_starts[sock_id]
        log.debug("Cleaned up flow_slow_starts for sock_id: {}".format(sock_id))
    
    if sock_id in flow_last_response_time:
        del flow_last_response_time[sock_id]
        log.debug("Cleaned up flow_last_response_time for sock_id: {}".format(sock_id))



def polling():
    while not cont.is_set():
        # Process any pending batch inferences
        process_batch_inferences()
        
        # Send periodic MeasureMsg requests
        send_periodic_measure_requests()
        
        # Check for and remove timeout flows
        check_and_remove_timeout_flows()
        
        if poller.poll_once() == False:
            # just sleep for a while (5ms to match batch processing interval)
            
            #print("CURRENT POLLER ACTION in polling:", poller.get_all_actions())
            time.sleep(0.005)


def main(args):
    # Initialize the DoubleTree model
    initialize_double_tree()
    
    # Initialize batch processing time
    global last_batch_time, last_measure_time, last_timeout_check
    last_batch_time = time.time()
    last_measure_time = time.time()
    last_timeout_check = time.time()
    
    # create the default env for spine
    env_flows.register_env(env_id)

    # recv new spine flow info and misc
    netlink_read_wrapper = partial(read_netlink_message, nl_sock)
    poller.add_action(
        Action(nl_sock, PollEvents.READ_ERR_FLAGS, callback=netlink_read_wrapper)
    )
    # print("CURRENT POLLER ACTION:", poller.get_all_actions())
    threading.Thread(target=polling).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--user",
        "-u",
        type=str,
        default="tianhan",
        help="the effective user after drop root privileges, we assume the same gid_name as uid_name",
    )
    parser.add_argument(
        "--alg",
        "-a",
        type=str,
        default="vanilla",
        help="kernel CC algorithm",
    )
    args = parser.parse_args()
    nl_sock = build_netlink_sock()
    drop_privileges(uid_name=args.user, gid_name=args.user)
    # after build netlink socket, we try to drop root privilege
    # assign netlink message sender
    kernel_cc = args.alg
    nl_send = getattr(msg_sender, "send_{}_message".format(kernel_cc)) 
    main(args)
