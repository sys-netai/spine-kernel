import os
import json
from ipc_socket import IPCSocket


# 假设你有一个 DoubleTree 类
from policycache import DoubleTree, TreeType

# 初始化 double tree
double_tree = DoubleTree(dt_path="decision_tree_online.pkl",
                         online_tree_type=TreeType.VFDT,
                         n_classes=[0, 1, 2])

IPC_PATH = "/tmp/inference_service_ipc"

# 如果 socket 文件已存在，先删除
if os.path.exists(IPC_PATH):
    os.remove(IPC_PATH)

# 启动 IPC 服务
ipc = IPCSocket()
ipc.bind(IPC_PATH)
ipc.listen()
print(f"[inference_service] Listening on {IPC_PATH}")

step_counter = 0

while True:
    print("waiting for client")
    client = ipc.accept()
    print("client connected")
    while True:
        try:
            msg = client.read()
            if not msg:
                break
            data = json.loads(msg)
            req_type = data.get("type", "inference")
            if req_type == "inference":
                state = data["state"]
                dt_action, vfdt_action = double_tree.predict(state)
                reply = {
                    "dt_action": int(dt_action[0]),
                    "vfdt_action": int(vfdt_action[0])
                }
                # print("inference: state:", state, "reply:", reply)
                client.write(json.dumps(reply))
            elif req_type == "update":
                state = data["state"]
                action = data["action"]
                # print("update: state:", state, "action:", action)
                double_tree.update_ot(state, action)
                client.write(json.dumps({"status": "updated"}))
            else:
                client.write(json.dumps({"error": "Unknown request type"}))
            # Increment step counter for every valid request
            step_counter += 1
            if step_counter % 500 == 0:
                double_tree.save("decision_tree_online.pkl", "vfdt_online.pkl")
                print("[inference_service] DoubleTree saved at step", step_counter)
                step_counter = 0
        except ConnectionResetError:
            print("Client disconnected unexpectedly")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON received: {e}")
            client.write(json.dumps({"error": "Invalid JSON format"}))
        except Exception as e:
            client.write(json.dumps({"error": str(e)}))
    client.close()