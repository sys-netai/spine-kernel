import pickle
import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from ding.policy import create_policy
from easydict import EasyDict
import os

from collections import deque

# use a enum for tree type 
class TreeType:
    VFDT = 0
    EFDT = 1
    HAT = 2
    DNN = 3
    
class DistillType:
    DNN = 0
    ONLINE = 1
    NONE = 2

class DoubleTree:
    """
    双树模型：
      - dt:   离线蒸馏的 sklearn 决策树
      - vfdt: skmultiflow HoeffdingTreeClassifier
    """
    def __init__(self,
                 dt_path: str = 'decision_tree_online.pkl',
                 n_classes: list = [0, 1],
                 online_tree_type: TreeType = TreeType.VFDT
                 ):
        # 1. 加载离线决策树
        # dt_path not existed
        self.dt_path = dt_path
        if (dt_path is None or dt_path == '') or not os.path.exists(dt_path):
            # create a new dt 
            self.dt = None
        else:
            with open(dt_path, 'rb') as f:
                self.dt = pickle.load(f)
        # 2. 根据 online_tree_type 创建对应的在线树
        self.online_tree_type = online_tree_type
        # 训练 VFDT 时需要知道所有可能的类别
        self.n_classes = n_classes
        self.last_printed_samples = 0  # 添加这行
        self.print_count = 0  # 添加这行
        self.episode_count = 0  # 添加这行
        self.initialize_online_tree()
        

    def save(self, dt_path, vfdt_path):
        with open(dt_path, 'wb') as f:
            # dump the dt and vfdt
            pickle.dump(self.dt, f)
        with open(vfdt_path, 'wb') as f:
            # Save both the online tree and learned_samples count
            save_data = {
                'online_tree': self.online_tree,
                'learned_samples': self.learned_samples
            }
            pickle.dump(save_data, f)

    def load(self, dt_path, vfdt_path):
        with open(dt_path, 'rb') as f:
            # load the dt and vfdt
            self.dt = pickle.load(f)
        with open(vfdt_path, 'rb') as f:
            # Load both the online tree and learned_samples count
            load_data = pickle.load(f)
            if isinstance(load_data, dict):
                # New format with learned_samples
                self.online_tree = load_data['online_tree']
                self.learned_samples = load_data['learned_samples']
            else:
                # Old format - just the online tree
                self.online_tree = load_data
                self.learned_samples = 0  # Default for old format

    def predict(self, state):
        """
        :param state: 一维特征向量，长度 = n_features_
        :return: (a_dt, a_vfdt)，都是 int
        """
        # 确保是 numpy 二维数组，shape=(1, n_features)
        x = np.array(state, dtype=np.float32).reshape(1, -1)
        if self.dt is None:
            a_dt = None
        else:
            a_dt = self.dt.predict(x)
        # 在线 Hoeffding 树预测
        a_ot = self.online_tree.predict(x)
        return a_dt, a_ot
    
    def predict_prob(self, state):
        """
        :param state: 一维特征向量，长度 = n_features_
        :return: (a_dt, a_vfdt)，都是 int
        """
        # 确保是 numpy 二维数组，shape=(1, n_features)
        x = np.array(state, dtype=np.float32).reshape(1, -1)
        if self.dt is None:
            a_dt = None
        else:
            a_dt = self.dt.predict_proba(x)
        # 在线 Hoeffding 树预测
        a_ot = self.online_tree.predict_proba(x)
        return a_dt, a_ot
    
    def update_ot(self, state, label):
        """
        用一个最新 (state, label) 样本增量训练 VFDT
        :param state:  1-D 特征向量
        :param label:  动作类别（0 / 1 / 2）
        """
        # if np.random.random() > 0.1:  # 0.2 = 1/5
        #     return
        # print(f"update_ot: state: {state}, label: {label}", "learned_samples: ", self.learned_samples)
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if type(label) == int:
            label = [label]
        self.online_tree.partial_fit(x, label, classes=self.n_classes)
        # Increment the learned_samples counter
        self.learned_samples += 1
        if self.learned_samples % 500 == 0:
            self.save_online_tree_snapshot()
        # print("learned samples number: ", self.learned_samples)

    def print_episode_samples(self):
        current_samples = self.learned_samples
        
        # 如果样本数没有变化，只打印前3次
        if current_samples == self.last_printed_samples:
            if self.print_count < 3:
                print(f"Online tree learned samples: {current_samples} (unchanged)")
                self.print_count += 1
                self.episode_count += 1
            return
        
        # 样本数有变化，重置计数并打印
        self.print_count = 0
        episode = self.episode_count // 3
        self.last_printed_samples = current_samples
        print(f"Online tree learned samples: {current_samples} (updated), episode: {episode}")

    def save_online_tree_snapshot(self):
        """
        保存在线树的快照，文件名格式为 online_tree_x，x是第x次保存
        保存到 src/model/tree 文件夹下
        """
        # 计算这是第几次保存（每500个样本保存一次）
        snapshot_number = self.learned_samples // 500
        
        # 设置保存目录为 src/model/tree
        # 从当前文件位置 (src/agent/policycache.py) 计算相对路径
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src/agent
        base_dir = os.path.dirname(current_dir)  # src
        model_dir = os.path.join(base_dir, "model", "tree")
        
        # 确保目录存在
        os.makedirs(model_dir, exist_ok=True)
        
        # 生成保存文件名
        snapshot_filename = f"online_tree_{snapshot_number}.pkl"
        snapshot_path = os.path.join(model_dir, snapshot_filename)
        # 保存在线树
        try:
            with open(snapshot_path, 'wb') as f:
                pickle.dump(self.online_tree, f)
            print(f"已保存在线树快照: {snapshot_path} (样本数: {self.learned_samples})")
        except Exception as e:
            print(f"保存在线树快照失败: {e}")

    def load_online_tree_snapshot(self, snapshot_number):
        """
        加载指定编号的在线树快照
        
        Args:
            snapshot_number: 快照编号（如1, 2, 3...）
        """
        # 设置加载目录
        current_dir = os.path.dirname(os.path.abspath(__file__))  # src/agent
        base_dir = os.path.dirname(current_dir)  # src
        model_dir = os.path.join(base_dir, "model", "tree")
        
        # 生成快照文件路径
        snapshot_filename = f"online_tree_{snapshot_number}.pkl"
        snapshot_path = os.path.join(model_dir, snapshot_filename)
        
        # 检查文件是否存在
        if not os.path.exists(snapshot_path):
            print(f"快照文件不存在: {snapshot_path}")
            return False
        
        # 加载快照
        try:
            with open(snapshot_path, 'rb') as f:
                load_data = pickle.load(f)
            
            if isinstance(load_data, dict):
                # 新格式
                self.online_tree = load_data['online_tree']
                self.learned_samples = load_data['learned_samples']
                loaded_snapshot_number = load_data.get('snapshot_number', 0)
                print(f"已加载在线树快照: {snapshot_path}")
                print(f"快照编号: {loaded_snapshot_number}, 样本数: {self.learned_samples}")
                return True
            else:
                # 旧格式兼容
                self.online_tree = load_data
                print(f"已加载在线树快照: {snapshot_path} (旧格式)")
                return True
                
        except Exception as e:
            print(f"加载在线树快照失败: {e}")
            return False
    
    def get_ot_depth(self):
        return self.online_tree.measure_tree_depth(), self.online_tree.measure_byte_size(), self.online_tree.get_model_description()
    
    def reload_distill_tree(self, dt_path):
        with open(dt_path, 'rb') as f:
            self.dt = pickle.load(f)
            
            
    def generate_new_instance(self):
        """
        Creates a new instance of the DoubleTree instance.
        """
        # Create a new instance of DoubleTree
        new_double_tree = DoubleTree(
            dt_path=self.dt_path,
            n_classes=self.n_classes,
            online_tree_type=self.online_tree_type
        )
        return new_double_tree      
    
    
    def initialize_online_tree(self):
        # reinitialize an online tree
        if self.online_tree_type == TreeType.VFDT:
            self.online_tree = HoeffdingTreeClassifier(
            grace_period=100,              # Minimum number of samples a node should observe before split attempt
            split_criterion='info_gain',   # Use information gain as the split criterion
            tie_threshold=0.05,            # Threshold for deciding ties between split candidates
            leaf_prediction='nba',         # Use Naive Bayes Adaptive for leaf prediction
            binary_split=True,             # Use binary splits for features
            split_confidence=1e-7,         # Confidence level for split decisions (lower = more splits)
            )
        elif self.online_tree_type == TreeType.EFDT:
            self.online_tree = ExtremelyFastDecisionTreeClassifier(
            grace_period=100,                # 每个节点至少需要看到n个数据点
            split_criterion='info_gain',    # 使用信息增益作为分裂标准
            tie_threshold=0.1,              # 调整特征分裂的阈值
            leaf_prediction='nba',          # 使用朴素贝叶斯作为叶节点的预测策略
            binary_split=True,              # 使用二分裂
            split_confidence=1e-7,          # 较低的分裂置信度
            nb_threshold=100                # 朴素贝叶斯预测至少需要100个样本
            )
        elif self.online_tree_type == TreeType.HAT:
            self.online_tree = HoeffdingAdaptiveTreeClassifier(
            grace_period=300,           # 每个节点见到多少样本后才考虑分裂
            split_confidence=1e-7,     # 分裂置信度，越小越容易分裂
            tie_threshold=0.025,        # 多特征差异<此阈值时视作平局，延迟分裂
            leaf_prediction='nba',     # 叶子预测方式：'nba' = Naive Bayes Adaptive
            nb_threshold=100,           # 叶节点样本数≥此值才启用 NB，<则多数类
            binary_split=True          # 二分裂（True）或多分裂（False）
            )   
        elif self.online_tree_type == TreeType.DNN:
            self.online_tree = OnlineMLP(input_dim=None, n_classes=self.n_classes)
        else:
            raise ValueError(f"Unsupported online_tree_type: {self.online_tree_type}")
        # Track the number of samples learned by the online tree
        self.learned_samples = 0

class OnlineStandardizer:
    """按样本流做滑动均值/方差标准化；对概念漂移更稳。"""
    def __init__(self, momentum=0.01, eps=1e-6):
        self.m = momentum
        self.eps = eps
        self.mean = None
        self.var = None

    def partial_fit(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if self.mean is None:
            self.mean = X.mean(axis=0)
            self.var  = X.var(axis=0) + self.eps
        else:
            mu = X.mean(axis=0)
            va = X.var(axis=0)
            # 指数滑动
            self.mean = (1 - self.m) * self.mean + self.m * mu
            self.var  = (1 - self.m) * self.var  + self.m * va + self.eps
        return self.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        return (X - self.mean) / np.sqrt(self.var)

class OnlineMLP:
    """
    改进点：
      1) 在线标准化 (OnlineStandardizer)
      2) 最近样本重放缓冲区 + 小批量训练
      3) 统一标签形状 (N,)
      4) 更稳健的优化器与超参：Adam(lr=3e-4) + 梯度裁剪 + label smoothing
      5) NaN/Inf 清洗
    """
    def __init__(self,
                 input_dim=None,
                 hidden_dim=32,
                 n_classes=3,
                 learning_rate=3e-4,
                 weight_decay=1e-4,
                 buffer_size=2048,
                 batch_size=32,
                 replay_per_step=1,   # 每次新样本到来抽几次 batch 训练
                 max_grad_norm=1.0):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_classes = len(n_classes) if isinstance(n_classes, list) else n_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.batch_size = batch_size
        self.replay_per_step = replay_per_step
        self.max_grad_norm = max_grad_norm

        self.model = None
        self.optimizer = None
        # 轻微的 label smoothing 有助于抗噪
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.is_fitted = False

        # 在线标准化 & 重放缓存
        self.scaler = OnlineStandardizer(momentum=0.01, eps=1e-6)
        self.buf_X = deque(maxlen=buffer_size)
        self.buf_y = deque(maxlen=buffer_size)

    def _init_model(self, input_dim):
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.LayerNorm(self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.n_classes)
        )
        self.optimizer = optim.AdamW(self.model.parameters(),
                                     lr=self.learning_rate,
                                     weight_decay=self.weight_decay)

    def _sanitize_X(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        # 统一处理 NaN/Inf
        return np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    def _train_on_batch(self, Xb: np.ndarray, yb: np.ndarray):
        X_tensor = torch.from_numpy(Xb.astype(np.float32))
        y_tensor = torch.from_numpy(yb.astype(np.int64))

        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(X_tensor)
        loss = self.criterion(out, y_tensor)
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        # 统一形状与类型
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        assert X.shape[0] == y.shape[0], "X 与 y 样本数量不一致"

        # 初始化模型
        if self.model is None:
            self._init_model(X.shape[1])

        # 清洗 + 在线标准化（更新统计量 & 返回归一化 X）
        X = self._sanitize_X(X)
        X = self.scaler.partial_fit(X)

        # 进缓存
        for i in range(X.shape[0]):
            self.buf_X.append(X[i])
            self.buf_y.append(y[i])

        # 训练：每来一批数据，抽 replay_per_step 次小批量优化
        if len(self.buf_X) >= max(self.batch_size, 8):
            for _ in range(self.replay_per_step):
                # 简单随机采样；如果严重不均衡，可在此做类别均衡采样
                idx = np.random.choice(len(self.buf_X), size=min(self.batch_size, len(self.buf_X)), replace=False)
                Xb = np.stack([self.buf_X[i] for i in idx])
                yb = np.asarray([self.buf_y[i] for i in idx], dtype=np.int64)
                self._train_on_batch(Xb, yb)

        self.is_fitted = True

    def predict(self, X):
        if self.model is None or not self.is_fitted:
            # 未训练好时，返回均匀类（比随机更稳）
            return np.zeros((len(np.atleast_2d(X))), dtype=np.int64)

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._sanitize_X(X)
        # 预测阶段也要用已学习到的均值方差做 transform
        X = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(X))
            pred = torch.argmax(out, dim=1).cpu().numpy()
        return pred

    def predict_proba(self, X):
        if self.model is None or not self.is_fitted:
            return np.ones((len(np.atleast_2d(X)), self.n_classes), dtype=np.float32) / self.n_classes

        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = self._sanitize_X(X)
        X = self.scaler.transform(X)

        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.from_numpy(X))
            prob = torch.softmax(out, dim=1).cpu().numpy()
        return prob

    # 下面三个只是为了与你原有接口兼容
    def measure_tree_depth(self):
        return len([m for m in self.model.modules() if isinstance(m, nn.Linear)]) if self.model else 0

    def measure_byte_size(self):
        if self.model is None: return 0
        p = sum(p.numel() * p.element_size() for p in self.model.parameters())
        b = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return p + b

    def get_model_description(self):
        if self.model is None:
            return "Uninitialized OnlineMLP"
        return f"OnlineMLP(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, n_classes={self.n_classes})"


class testDNN:
    def __init__(self, model_path: str, exp_config: dict):
        """
        Args:
            model_path  : ckpt_best_xxxx.pth.tar 路径
            exp_config  : 与训练时相同的 policy 配置（dict/ EasyDict）
        """
        # 1) 创建策略骨架
        cfg = EasyDict(exp_config)              # 转成 EasyDict 方便
        self._policy = create_policy(cfg.policy, enable_field=['eval'])

        # 2) 加载权重
        state_dict = torch.load(model_path, map_location='cpu')
        self._policy.eval_mode.load_state_dict(state_dict)

        # 3) 切 eval 模式（禁用 dropout / BN 更新）
        self.eval_policy= self._policy.eval_mode
    @torch.no_grad()
    def predict(self, state):
        """
        Args:
            state : list / np.ndarray, shape=(STATE_DIM,)
        Returns:
            action(int) : 0 / 1 / 2
        """
        # 1) 转成 (1, dim) float32 numpy
        obs = np.array(state, dtype=np.float32).reshape(1, -1)

        obs = {0: torch.Tensor(obs)}

        # 3) 推理
        policy_output = self.eval_policy.forward(obs)[0]

        # 4) 取动作
        dnn_action = policy_output["action"].detach().cpu().numpy()[0]
        dnn_action = np.array(dnn_action, dtype=np.int64)
        return dnn_action
