import os
import json
from ipc_socket import IPCSocket
import threading
import numpy as np

# 假设你有一个 DoubleTree 类
from policycache import DoubleTree, TreeType
from poller import Action, Poller, ReturnStatus, PollEvents

# cont status of polling
cont = threading.Event()
poller = Poller()


# delete the current model decision_tree_online.pkl
if os.path.exists("decision_tree_online.pkl"):
    os.remove("decision_tree_online.pkl")
 
# 初始化 double tree
double_tree = DoubleTree(dt_path="decision_tree_online.pkl",
                         online_tree_type=TreeType.HAT,
                         n_classes=[0, 1])

IPC_PATH = "/tmp/inference_service_ipc"

## 如果 socket 文件已存在，先删除
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
                dt_action_prob, vfdt_action_prob = double_tree.predict_prob(state)
                # TODO:得到dt_action和vfdt_action，dt_action_prob和vfdt_action_prob
                # 这4个if就可以不用管开不开RUN_STATIC和RUN_DYNAMIC了，反正后面会赋分数0的
                if dt_action_prob is None :
                    dt_action_prob = np.array([[0.5, 0.5]])
                if vfdt_action_prob is None :
                    vfdt_action_prob = np.array([[0.5, 0.5]])
                if len(vfdt_action_prob[0]) < 2 or sum(vfdt_action_prob[0]) < 0.99:
                    vfdt_action_prob = np.array([[0.5, 0.5]])
                if len(dt_action_prob[0]) < 2 or sum(dt_action_prob[0]) < 0.99:
                    dt_action_prob = np.array([[0.5, 0.5]])
                dt_action = np.random.choice(np.arange(len(dt_action_prob[0])), p=dt_action_prob[0])
                vfdt_action = np.random.choice(np.arange(len(vfdt_action_prob[0])), p=vfdt_action_prob[0])
                #TODO:状态动作对齐，这里需要修改
                reply = {
                        "dt_action": int(dt_action),
                        "vfdt_action": int(vfdt_action),
                        "dt_action_prob": dt_action_prob[0].tolist(),
                        "vfdt_action_prob": vfdt_action_prob[0].tolist()
                }
                # print("inference: state:", state, "reply:", reply)
                client.write(json.dumps(reply))
            elif req_type == "update":
                state = data["state"]
                action = data["action"]
                print("update: state:", state, "action:", action)
                if(action == 0 or action == 1):
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