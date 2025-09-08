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
                 dt_path: str = 'decision_tree.pkl',
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
        self.initialize_online_tree()
        # 训练 VFDT 时需要知道所有可能的类别
        self.n_classes = n_classes

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
        # print(f"update_ot: state: {state}, label: {label}", "learned_samples: ", self.learned_samples)
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if type(label) == int:
            label = [label]
        self.online_tree.partial_fit(x, label, classes=self.n_classes)
        # Increment the learned_samples counter
        self.learned_samples += 1
        # print("learned samples number: ", self.learned_samples)

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
            grace_period=500,              # Minimum number of samples a node should observe before split attempt
            split_criterion='info_gain',   # Use information gain as the split criterion
            tie_threshold=0.05,            # Threshold for deciding ties between split candidates
            leaf_prediction='nba',         # Use Naive Bayes Adaptive for leaf prediction
            binary_split=True,             # Use binary splits for features
            split_confidence=1e-6,         # Confidence level for split decisions (lower = more splits)
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
            grace_period=100,           # 每个节点见到多少样本后才考虑分裂
            split_confidence=1e-7,     # 分裂置信度，越小越容易分裂
            tie_threshold=0.05,        # 多特征差异<此阈值时视作平局，延迟分裂
            leaf_prediction='nba',     # 叶子预测方式：'nba' = Naive Bayes Adaptive
            nb_threshold=30,           # 叶节点样本数≥此值才启用 NB，<则多数类
            binary_split=True          # 二分裂（True）或多分裂（False）
            )   
        elif self.online_tree_type == TreeType.DNN:
            self.online_tree = OnlineMLP(input_dim=None, n_classes=self.n_classes)
        else:
            raise ValueError(f"Unsupported online_tree_type: {self.online_tree_type}")
        # Track the number of samples learned by the online tree
        self.learned_samples = 0

class OnlineMLP:
    """
    Simple MLP for online learning that mimics the interface of skmultiflow classifiers
    """
    def __init__(self, input_dim=None, hidden_dim=64, n_classes=3, learning_rate=0.1):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Handle n_classes being a list or int
        self.n_classes = len(n_classes) if isinstance(n_classes, list) else n_classes
        self.learning_rate = learning_rate
        
        # Will be initialized on first call to partial_fit
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.is_fitted = False
        
    def _init_model(self, input_dim):
        """Initialize the model when we know the input dimension"""
        self.input_dim = input_dim
        self.model = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.n_classes)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Incrementally fit the model with a batch of samples
        """
        # Convert inputs to numpy if needed
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)
            
        # Initialize model if not done yet
        if self.model is None:
            self._init_model(X.shape[1])
            
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Training step
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
        
        self.is_fitted = True
        
    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        if not self.is_fitted or self.model is None:
            # If not fitted, return random predictions
            return np.random.randint(0, self.n_classes, size=len(X))
            
        # Convert to numpy if needed
        if isinstance(X, list):
            X = np.array(X)
            
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy()
            
        return predictions
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X
        """
        if not self.is_fitted or self.model is None:
            # If not fitted, return uniform probabilities
            return np.ones((len(X), self.n_classes)) / self.n_classes
            
        # Convert to numpy if needed
        if isinstance(X, list):
            X = np.array(X)
            
        # Convert to torch tensor
        X_tensor = torch.FloatTensor(X)
        
        # Prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1).numpy()
            
        return probabilities
    
    def measure_tree_depth(self):
        """Return depth information for compatibility"""
        return len([m for m in self.model.modules() if isinstance(m, nn.Linear)])
    
    def measure_byte_size(self):
        """Return model size information for compatibility"""
        if self.model is None:
            return 0
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        return param_size + buffer_size
    
    def get_model_description(self):
        """Return model description for compatibility"""
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

'''from policycache import testDNN
dnn = testDNN(
    model_path='train/config/serial/single_json_nn7_seed9/ckpt/ckpt_best_28000.pth.tar',
    exp_config=exp_config
)

action = dnn.predict(s0)                # s0 为 transform_state 输出
print('DNN action =', action)'''

# Example usage of DNN TreeType:
'''
from policycache import DoubleTree, TreeType

# Create a DoubleTree with DNN online learning
double_tree = DoubleTree(
    dt_path=None,  # No offline tree
    n_classes=[0, 1, 2],  # Three action classes
    online_tree_type=TreeType.DNN
)

# Online learning examples
import numpy as np

# Sample state and action data
state1 = np.random.rand(10)  # 10-dimensional state
action1 = 1  # Action class

state2 = np.random.rand(10)
action2 = 0

# Update the online MLP with new samples
double_tree.update_ot(state1, action1)
double_tree.update_ot(state2, action2)

# Make predictions
predicted_actions = double_tree.predict(state1)
print(f"Predicted actions: {predicted_actions}")  # (offline_tree_action, online_mlp_action)

# Get model information
depth, size, description = double_tree.get_ot_depth()
print(f"Model depth: {depth}, Size: {size} bytes")
print(f"Description: {description}")
'''