from enum import Enum
import os
import stat
import sys
import json
import time
import argparse
import threading
import functools
from functools import partial

# flow id comes from the unix socket
# sock id comes from the kernel
# port id comes from the env, which is coherent as the flows are initizlied by the env

from logger import logger as log
from message import *
from netlink import Netlink
from ipc_socket import IPCSocket
from spine_flow import Flow, ActiveFlowMap, EnvFlows
from poller import Action, Poller, ReturnStatus, PollEvents
from helper import drop_privileges
import msg_sender

# communication between spine-user-space and kernel
nl_sock = None
# communication between spine-user-space and Env
unix_sock = None
# the unix socks with key as sock_id come from the kernel
client_from_sock = dict()
# kernel cc alg
kernel_cc = None
# message sender
nl_send = None
# cont status of polling
cont = threading.Event()
env_flows = EnvFlows()
poller = Poller()


class UnixMessageType(Enum):
    INIT = 0  # env initialization
    START = 1  # episode start
    END = 2  # episode end
    ALIVE = 3  # alive status
    OBSERVE = 4  # observe the world
    TERMINATE = 5  # terminate the env
    MEASURE = 6


def build_unix_sock(unix_file):
    if os.path.exists(unix_file):
        log.debug("{} already exists, remove it".format(unix_file))
        os.remove(unix_file)
    sock = IPCSocket()
    log.debug("UNIX IPC file: {}".format(unix_file))
    sock.bind(unix_file)
    sock.set_noblocking()
    sock.listen()
    log.info("Spine is listening for flows from env at {}".format(unix_file))
    return sock


def build_netlink_sock():
    sock = Netlink()
    sock.add_mc_group()
    return sock


def read_netlink_message(nl_sock: Netlink):
    hdr_raw = nl_sock.next_msg()
    if hdr_raw == None:
        return ReturnStatus.Cancel
    hdr = SpineMsgHeader()
    if hdr.from_raw(hdr_raw) == None:
        log.error("Failed to parse netlink header")

    if hdr.type == NL_CREATE:
        msg = CreateMsg()
        msg.from_raw(hdr_raw[hdr.hdr_len :])
        flow = Flow().from_create_msg(msg, hdr)
        # first find env
        dst_endpoint = (flow.dst_ip, flow.dst_port)
        env_id = env_flows.dst_endpoint_to_env_id.get(dst_endpoint, None)
        if env_id == None:
            # log.warn("unknown dst_port: {}".format(flow.dst_port))
            return ReturnStatus.Continue
        # register new flow
        active_flow_map = env_flows.get_env_flows(env_id)
        if active_flow_map == None:
            log.warn("env: {} has not registered".format(env_id))
            return ReturnStatus.Continue
        active_flow_map.add_flow_with_sockId(flow)
        # cache sockID with envid
        env_flows.bind_sock_id_to_env(flow.sock_id, env_id)
        # print ("add flow with sock_id: ", flow.sock_id)
        # print ("add flow with flow_id: ", active_flow_map.get_flowId_by_sockId(flow.sock_id))
        # print ("add flow with dst_port: ", flow.dst_port)
        # print("After the association: ", env_id, active_flow_map.get_all_flow_ids(), active_flow_map, active_flow_map.sock_id_to_flow_id, active_flow_map.dst_port_to_sock_id)
        return ReturnStatus.Continue
    elif hdr.type == NL_READY:
        log.info("Spine kernel is ready!!")
    elif hdr.type == NL_MEASURE:
        sock_id = hdr.sock_id
        msg = MeasureMsg()
        msg.from_raw(hdr_raw[hdr.hdr_len :])
        msg_to_unix = {}
        msg_to_unix["type"] = UnixMessageType.MEASURE.value
        msg_to_unix["data"] = msg.data
        msg_to_unix["request_id"] = msg.request_id
        # print the time, flow_id, and data
        # print ("read netlink message: ", time.time(), msg.data)
        # first find env
        env_id = env_flows.sock_id_to_env_id.get(sock_id, None)
        if env_id == None:
            log.warn("unknown dst_port: {}".format(flow.dst_port))
            return ReturnStatus.Continue
        active_flow_map = env_flows.get_env_flows(env_id)
        flow_id = active_flow_map.get_flowId_by_sockId(sock_id)
        # print("get a new measure from flow {}!".format(flow_id))
        # print("get from sock_id: ", sock_id, "flow id map: ", active_flow_map.sock_id_to_flow_id, "env id: ", env_id)
        if flow_id == None:
            print("want to get from sock_id: ", sock_id, "flow id map: ", active_flow_map, active_flow_map.sock_id_to_flow_id, "env id: ", env_id)
            # print("Warning: a none flow_id")
            return ReturnStatus.Continue
        # msg_to_unix["flow_id"] = flow_id
        client_sock = client_from_sock[sock_id]
        sent = client_sock.write(json.dumps(msg_to_unix))
        if sent == 0:
            print("Nothing sent...")
        # print("write to unix.")
    elif hdr.type == NL_RELEASE:
        # flow release
        sock_id = hdr.sock_id
        # we just remove the cached items
        env_flows.release_sock_id_to_env(sock_id)
        # env has been deregistered, do nothing
    return ReturnStatus.Continue


def read_unix_message(unix_sock: IPCSocket):
    raw = unix_sock.read(header=True)
    if raw == None:
        return ReturnStatus.Cancel
    data = json.loads(raw)
    env_id = str(data["env_id"])
    flow_id = int(data["flow_id"])
    msg_type = data["type"]
    # associate spine-kernel sock id with flow_id
    active_flow_map = env_flows.get_env_flows(env_id)
    if active_flow_map == None:
        log.warn("env {} has not registered.".format(env_id))
        return ReturnStatus.Continue

    if msg_type == UnixMessageType.START.value:
        dst_ip = int(data["dst_ip"])
        dst_port = int(data["dst_port"])
        # we also need to record the corresponce of env_id and dst_endpoint
        env_flows.bind_endpoint_to_env(dst_ip, dst_port, env_id)
        active_flow_map.add_flow_with_dst_endpoint(dst_ip, dst_port, flow_id)
        return ReturnStatus.Continue
    elif msg_type == UnixMessageType.TERMINATE.value:
        active_flow_map.remove_all_env_flows()
        # deregister env
        env_flows.release_env(env_id)
        return ReturnStatus.Cancel
    elif msg_type == UnixMessageType.END.value:
        # we need the dst_endpoint to remove the cache
        sock_id = active_flow_map.get_sockId_by_flowId(flow_id)
        dst_endpoint = active_flow_map.remove_flow_by_flowId(flow_id)
        # remove cached items
        if dst_endpoint:
            env_flows.release_endpoint_to_env(dst_endpoint[0], dst_endpoint[1])
        return ReturnStatus.Continue
    elif msg_type == UnixMessageType.MEASURE.value:
        # print the time, flow_id 
        # print ("read unix message: ", time.time(), flow_id)
        # we need the dsr_port id to remove the cache
        sock_id = active_flow_map.get_sockId_by_flowId(flow_id)
        client_from_sock[sock_id] = unix_sock # cache the client socket, however, the current unix socket is unique for an env.
        # print("get a new measure request from unix! update the sock dict:", sock_id, unix_sock)
        assert nl_send != None
        if data["request_id"] is None:
            return ReturnStatus.Continue
        # request id is used for aligning measurements request and reply at the env
        # We don't need request-id to align request and reply if the neo_get_state is blocking.
        
        if sock_id == None:
            print("Warning: a none sock_id at time {} during unix reading".format(time.time()))
            return ReturnStatus.Continue
        nl_send(data["request_id"], nl_sock, sock_id, msg_type=NL_MEASURE)
        return ReturnStatus.Continue
    elif msg_type == UnixMessageType.OBSERVE.value:
        # print("get a new observe request from unix!")
        # send to all flows
        for flow_id in active_flow_map.get_all_flow_ids():
            print("flow: ", flow_id)
            sock_id = active_flow_map.get_sockId_by_flowId(flow_id)
            if sock_id == None:
                print("Warning: a none sock_id when observing, flow id:", flow_id)
                return ReturnStatus.Continue
            client_from_sock[sock_id] = unix_sock
            assert nl_send != None
            if data["request_id"] is None:
                return ReturnStatus.Continue
            # print("send to flow: ", flow_id)
            nl_send(data["request_id"], nl_sock, sock_id, msg_type=NL_MEASURE)
        return ReturnStatus.Continue
    # message should be ALIVE
    if msg_type != UnixMessageType.ALIVE.value:
        log.error("Incorrect message type: {}".format(msg_type))
        return ReturnStatus.Cancel
    # lookup sock id by flow_id
    sock_id = active_flow_map.get_sockId_by_flowId(flow_id)
    if sock_id == None:
        # log.warn("unknown flow id: {}".format(flow_id))
        return ReturnStatus.Continue

    # spine semantics: None means no action is need
    if data["action"] is None:
        return ReturnStatus.Continue

    assert nl_send != None
    nl_send(data["action"], nl_sock, sock_id)
    return ReturnStatus.Continue


def accept_unix_conn(unix_sock: IPCSocket, poller: Poller):
    client: IPCSocket = unix_sock.accept()
    # deal with init message
    message = client.read()
    message = json.loads(message)
    info = int(message.get("type", -1))
    if info != UnixMessageType.INIT.value:
        log.error("Incorrect message type: {}, ignore this".format(info))
        return ReturnStatus.Continue
    # accept new conn and register to poller
    env_id = str(message["env_id"])
    log.info(
        "Spine get connection from Env, env_id is {}, new ipc fd: {}".format(
            env_id, client.fileno()
        )
    )
    # register to env
    env_flows.register_env(env_id)
    client.set_noblocking()
    # unix_sock: recv updated parameters and relay to nl_sock
    unix_read_wrapper = partial(read_unix_message, client)
    poller.add_action(
        Action(client, PollEvents.READ_ERR_FLAGS, callback=unix_read_wrapper)
    )
    return ReturnStatus.Continue


def polling():
    while not cont.is_set():
        if poller.poll_once() == False:
            # just sleep for a while (10ms)
            time.sleep(0.01)


def main(args):
    # register accept for unix socket
    listen_callback = partial(accept_unix_conn, unix_sock, poller)
    poller.add_action(
        Action(
            unix_sock,
            PollEvents.READ_ERR_FLAGS,
            callback=listen_callback,
        )
    )

    # recv new spine flow info and misc
    netlink_read_wrapper = partial(read_netlink_message, nl_sock)
    poller.add_action(
        Action(nl_sock, PollEvents.READ_ERR_FLAGS, callback=netlink_read_wrapper)
    )
    threading.Thread(target=polling).run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    default_unix_file = "/tmp/spine_ipc"
    parser.add_argument(
        "--ipc",
        "-i",
        type=str,
        default=None,
        help="IPC communication between Env and Spine controller",
    )
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
    # build communication sockets
    if args.ipc is None:
        args.ipc = "{}_{}".format(default_unix_file, args.user)
    unix_sock = build_unix_sock(args.ipc)
    # mod: 775
    os.chmod(args.ipc, stat.S_IRWXU | stat.S_IRWXG | stat.S_IROTH | stat.S_IXOTH)
    # assign netlink message sender
    kernel_cc = args.alg
    nl_send = getattr(msg_sender, "send_{}_message".format(kernel_cc))
    main(args)
