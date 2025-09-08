import os
import json
import threading
import time
import numpy as np
from functools import partial

from ipc_socket import IPCSocket
from policycache import DoubleTree, TreeType
from poller import Action, Poller, ReturnStatus, PollEvents


# ---------- RWLock 实现 ----------
class RWLock:
    """简单的读写锁：允许多个读者并发，写者独占"""
    def __init__(self):
        self._readers = 0
        self._readers_lock = threading.Lock()
        self._writers_lock = threading.Lock()

    def acquire_read(self):
        with self._readers_lock:
            self._readers += 1
            if self._readers == 1:
                self._writers_lock.acquire()

    def release_read(self):
        with self._readers_lock:
            self._readers -= 1
            if self._readers == 0:
                self._writers_lock.release()

    def acquire_write(self):
        self._writers_lock.acquire()

    def release_write(self):
        self._writers_lock.release()


# ---------- Global setup ----------
cont = threading.Event()
poller = Poller()
last_seen = {}  # fd -> last activity timestamp
IDLE_TIMEOUT = 30  # seconds

# Locks
rwlock = RWLock()
last_seen_lock = threading.Lock()

double_tree = DoubleTree(
    dt_path="dt_tree.pkl",
    online_tree_type=TreeType.HAT,
    n_classes=[0, 1],
)

IPC_PATH = "/tmp/inference_service_ipc"
if os.path.exists(IPC_PATH):
    os.remove(IPC_PATH)

ipc = IPCSocket()
ipc.bind(IPC_PATH)
ipc.set_noblocking()
ipc.listen()
print(f"[inference_service] Listening on {IPC_PATH}")

step_counter = 0


# ---------- Helpers ----------
def cleanup_client(client: IPCSocket):
    """Remove client from poller and close it."""
    fd = client.fileno()
    with last_seen_lock:
        last_seen.pop(fd, None)
    try:
        print(f"Cleaning up client fd {fd}")
        poller.remove_action(fd)
    except Exception as e:
        print(f"Poller remove error for fd {fd}: {e}")
    try:
        client.close()
    except Exception as e:
        print(f"Client close error for fd {fd}: {e}")


# ---------- Core logic ----------
def handle_client_message(client: IPCSocket):
    global step_counter
    fd = client.fileno()
    try:
        msg = client.read(header=True)
        if msg is None:
            # client disconnected
            cleanup_client(client)
            return ReturnStatus.Cancel

        with last_seen_lock:
            last_seen[fd] = time.time()  # heartbeat update

        data = json.loads(msg)
        req_type = data.get("type", "inference")

        if req_type == "inference":
            state = data["state"]
            rwlock.acquire_read()
            try:
                dt_action_prob, vfdt_action_prob = double_tree.predict_prob(state)
            finally:
                rwlock.release_read()

            # sanity checks
            if dt_action_prob is None:
                dt_action_prob = np.array([[0.5, 0.5]])
            if vfdt_action_prob is None:
                vfdt_action_prob = np.array([[0.5, 0.5]])
            if len(vfdt_action_prob[0]) < 2 or sum(vfdt_action_prob[0]) < 0.99:
                vfdt_action_prob = np.array([[0.5, 0.5]])
            if len(dt_action_prob[0]) < 2 or sum(dt_action_prob[0]) < 0.99:
                dt_action_prob = np.array([[0.5, 0.5]])

            reply = {
                "dt_action_prob": dt_action_prob[0].tolist(),
                "vfdt_action_prob": vfdt_action_prob[0].tolist(),
            }
            client.write(json.dumps(reply))

        elif req_type == "update":
            state = data["state"]
            action = data["action"]
            rwlock.acquire_write()
            try:
                double_tree.update_ot(state, action)
            finally:
                rwlock.release_write()
            client.write(json.dumps({"status": "updated"}))
            step_counter += 1

        elif req_type == "alive":  # heartbeat only
            client.write(json.dumps({"status": "alive"}))

        else:
            client.write(json.dumps({"error": "Unknown request type"}))

        # Step counter
        if step_counter % 500 == 0:
            rwlock.acquire_write()
            try:
                double_tree.save("dt_tree.pkl", "online_tree.pkl")
            finally:
                rwlock.release_write()
            step_counter += 1 # otherwise it keep saving until the next update.
            print("[inference_service] DoubleTree saved at step", step_counter)

    except json.JSONDecodeError as e:
        print(f"Invalid JSON received from fd {fd}: {e}")
        client.write(json.dumps({"error": "Invalid JSON format"}))
        return ReturnStatus.Continue
    except Exception as e:
        print(f"Error handling client fd {fd}: {e}")
        cleanup_client(client)
        return ReturnStatus.Cancel

    return ReturnStatus.Continue


def accept_client(ipc: IPCSocket, poller: Poller):
    try:
        client = ipc.accept()
        if client is None:
            return ReturnStatus.Continue
        print(f"New client connected: fd {client.fileno()}")
        client.set_noblocking()
        fd = client.fileno()
        with last_seen_lock:
            last_seen[fd] = time.time()
        poller.add_action(
            Action(
                client,
                PollEvents.READ_ERR_FLAGS,
                callback=partial(handle_client_message, client),
            )
        )
    except Exception as e:
        print(f"Error accepting client: {e}")
    return ReturnStatus.Continue


# ---------- Threads ----------
def polling():
    while not cont.is_set():
        if not poller.poll_once():
            time.sleep(0.002)


def watchdog():
    """Close idle clients without heartbeat for > IDLE_TIMEOUT seconds."""
    while not cont.is_set():
        now = time.time()
        with last_seen_lock:
            stale = [
                fd for fd, last in last_seen.items()
                if now - last > IDLE_TIMEOUT
            ]
        for fd in stale:
            print(f"Client fd {fd} timed out (idle > {IDLE_TIMEOUT}s)")
            try:
                print("Watchdog cleaning up fd:", fd)
                actions = poller.get_action_by_fd(fd)
                if len(actions) == 0:
                    with last_seen_lock:
                        last_seen.pop(fd, None)
                    continue
                for action in actions:
                    cleanup_client(action.sock)
            except Exception as e:
                print(f"Watchdog cleanup error for fd {fd}: {e}")
        time.sleep(5)


def main():
    # register listening socket
    listen_callback = partial(accept_client, ipc, poller)
    poller.add_action(
        Action(ipc, PollEvents.READ_ERR_FLAGS, callback=listen_callback)
    )

    # start polling + watchdog in background threads
    threading.Thread(target=polling, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()

    # block main thread
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        cont.set()
        print("Shutting down inference service")


if __name__ == "__main__":
    main()