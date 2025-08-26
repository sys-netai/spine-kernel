import pickle
import numpy as np
from skmultiflow.trees import HoeffdingTreeClassifier
from skmultiflow.trees import ExtremelyFastDecisionTreeClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
import torch
import numpy as np
from ding.policy import create_policy
from easydict import EasyDict
import os

# use a enum for tree type 
class TreeType:
    VFDT = 0
    EFDT = 1
    HAT = 2
    
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
                 n_classes: list = [0, 1, 2],
                 online_tree_type: TreeType = TreeType.VFDT
                 ):
        # 1. 加载离线决策树
        # dt_path not existed
        if (dt_path is None or dt_path == '') or not os.path.exists(dt_path):
            # create a new dt 
            self.dt = None
        else:
            with open(dt_path, 'rb') as f:
                self.dt = pickle.load(f)
        # 2. 根据 online_tree_type 创建对应的在线树
        self.online_tree_type = online_tree_type
        if online_tree_type == TreeType.VFDT:
            self.online_tree = HoeffdingTreeClassifier(
            grace_period=500,
            split_criterion='info_gain',
            tie_threshold=0.05,
            leaf_prediction='nba',
            binary_split=True,
            split_confidence=1e-6,
            )
        elif online_tree_type == TreeType.EFDT:
            self.online_tree = ExtremelyFastDecisionTreeClassifier(
            grace_period=100,                # 每个节点至少需要看到50个数据点
            split_criterion='info_gain',    # 使用信息增益作为分裂标准
            tie_threshold=0.1,              # 调整特征分裂的阈值
            leaf_prediction='nba',          # 使用朴素贝叶斯作为叶节点的预测策略
            binary_split=True,              # 使用二分裂
            split_confidence=1e-7,          # 较低的分裂置信度
            nb_threshold=100                # 朴素贝叶斯预测至少需要100个样本
            )
        elif online_tree_type == TreeType.HAT:
            self.online_tree = HoeffdingAdaptiveTreeClassifier(
            grace_period=100,           # 每个节点见到多少样本后才考虑分裂
            split_confidence=1e-7,     # 分裂置信度，越小越容易分裂
            tie_threshold=0.05,        # 多特征差异<此阈值时视作平局，延迟分裂
            leaf_prediction='nba',     # 叶子预测方式：'nba' = Naive Bayes Adaptive
            nb_threshold=30,           # 叶节点样本数≥此值才启用 NB，<则多数类
            binary_split=True          # 二分裂（True）或多分裂（False）
            )
        else:
            raise ValueError(f"Unsupported online_tree_type: {online_tree_type}")
        # 训练 VFDT 时需要知道所有可能的类别
        self.n_classes = n_classes


    def save(self, dt_path, vfdt_path):
        with open(dt_path, 'wb') as f:
            # dump the dt and vfdt
            pickle.dump(self.dt, f)
        with open(vfdt_path, 'wb') as f:
            pickle.dump(self.online_tree, f)

    def load(self, dt_path, vfdt_path):
        with open(dt_path, 'rb') as f:
            # load the dt and vfdt
            self.dt = pickle.load(f)
        with open(vfdt_path, 'rb') as f:
            self.online_tree = pickle.load(f)

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
        x = np.asarray(state, dtype=np.float32).reshape(1, -1)
        if type(label) == int:
            label = [label]
        self.online_tree.partial_fit(x, label)

    def get_ot_depth(self):
        return self.online_tree.measure_tree_depth(), self.online_tree.measure_byte_size(), self.online_tree.get_model_description()
    
    def reload_distill_tree(self, dt_path):
        with open(dt_path, 'rb') as f:
            self.dt = pickle.load(f)

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