# 导入命令行参数解析模块，用于处理脚本运行时的命令行参数
import argparse
# 导入路径处理模块，用于处理文件路径和目录路径操作
import pathlib
# 导入类型注解模块，用于定义函数参数和返回值的类型
from typing import List, Tuple, Union

# 导入PyTorch深度学习框架的核心模块
import torch
# 导入PyTorch的神经网络模块，包含各种层和激活函数
import torch.nn as nn
# 导入RDKit化学信息学库中的分子对象类型
from rdkit.Chem.rdchem import Mol
# 导入PyTorch的张量类型别名
from torch import Tensor
# 导入PyTorch Geometric的批处理数据和数据对象类
from torch_geometric.data import Batch, Data
# 导入torch_scatter库的聚合函数，用于在图神经网络中进行节点聚合操作
from torch_scatter import scatter_mean, scatter_sum

# 从本地feature模块导入数据结构和分子处理函数
from .feature import PairData, from_mol, from_smiles
# 从本地layers模块导入各种神经网络层的实现
from .layers import AMPN, FMPN, FeedForward, MPNNEmbedding

# 注释掉的导入，可能是用于属性预测的模块
# from .property import PROPERTIES

# 定义SMILES字符串的类型别名，SMILES是化学分子的字符串表示法
SMILES = str
# 定义属性字符串的类型别名，用于表示分子的各种属性
PROPERTY = str

# 定义一个空行，用于代码分隔和可读性


# 定义DeepBioisostere深度学习模型类，继承自PyTorch的nn.Module基类
class DeepBioisostere(nn.Module):
    """
    Fragment-level structural modification model to explore bioisosteres.
    You can use this model by one of the following codes:
    1. model = FragModNet(<args>)
    2. model = FragModNet.from_trained_model(<path_to_model>, <args>)
    """

    # 定义模型的默认参数字典，包含所有可配置的超参数和设置
    default_args = {
        # 模型权重文件的路径，默认为"Not restored"表示未加载预训练模型
        "path_to_model": "Not restored",
        # 分子节点（原子）的特征维度，每个原子有49个特征
        "mol_node_features": 49,
        # 分子边（化学键）的特征维度，每个化学键有12个特征
        "mol_edge_features": 12,
        # 分子节点隐藏层的维度，用于神经网络的中间表示
        "mol_node_hid_dim": 128,
        # 分子边隐藏层的维度，用于边特征的神经网络表示
        "mol_edge_hid_dim": 128,
        # 分子嵌入层的数量，控制消息传递网络的深度
        "mol_num_emb_layer": 4,
        # 片段节点的特征维度，每个片段节点有66个特征
        "frag_node_features": 66,
        # 片段边的特征维度，每个片段边有12个特征
        "frag_edge_features": 12,
        # 片段节点隐藏层的维度
        "frag_node_hid_dim": 128,
        # 片段边隐藏层的维度
        "frag_edge_hid_dim": 128,
        # 片段嵌入层的数量，控制片段消息传递网络的深度
        "frag_num_emb_layer": 4,
        # 位置评分模块的隐藏层维度，用于预测分子中可修改位置的得分
        "position_score_hid_dim": 128,
        # 修改位置评分网络的层数
        "num_mod_position_score_layer": 3,
        # 片段评分模块的隐藏层维度，用于评估插入片段的合适性
        "frag_score_hid_dim": 128,
        # 片段评分网络的层数
        "num_frag_score_layer": 3,
        # 附着位点评分模块的隐藏层维度，用于预测片段与分子的连接位点
        "attach_score_hid_dim": 128,
        # 附着位点评分网络的层数
        "num_attach_score_layer": 3,
        # 片段级消息传递的层数，用于片段之间的信息交换
        "frag_message_passing_num_layer": 2,
        # Dropout概率，用于防止过拟合，0.0表示不使用dropout
        "dropout": 0.0,
        # 负采样的数量，用于训练时的对比学习
        "num_neg_sample": 20,
        # 是否使用条件嵌入，True表示使用属性信息作为条件
        "conditioning": True,
        # 使用的分子属性列表，默认为空
        "properties": [],
        # 属性的维度，每个属性的特征维度
        "properties_dim": 1,
        # 计算设备，默认使用CPU
        "device": torch.device("cpu"),
    }

    # 定义模型的初始化函数，接受可选的参数命名空间
    def __init__(self, args: argparse.Namespace = None):
        # 模型初始化阶段的注释
        # 调用父类nn.Module的初始化方法
        super().__init__()
        # 初始化默认设置，将默认参数赋值给当前实例
        self._init_default_setting()
        # 如果传入了参数，则用传入的参数覆盖默认参数
        if args:
            # 创建空字典存储模型相关参数
            model_args = dict()
            # 遍历args中的所有属性
            for key, value in vars(args).items():
                # 只保留在当前实例字典中存在的键值对
                if key in self.__dict__:
                    model_args[key] = value
            # 用筛选后的参数更新当前实例的属性
            self.__dict__.update(model_args)
        # 如果properties为None，则初始化为空列表
        if self.properties is None:
            self.properties = []

        # 第1部分：消息传递网络的定义
        # 创建原子嵌入层，用于处理分子中原子级别的信息
        atom_embedding = MPNNEmbedding(
            # 原子节点特征的输入维度
            node_feature_dim=self.mol_node_features,
            # 化学键边特征的输入维度
            edge_feature_dim=self.mol_edge_features,
            # 原子节点隐藏层维度
            node_hidden_dim=self.mol_node_hid_dim,
            # 化学键边隐藏层维度
            edge_hidden_dim=self.mol_edge_hid_dim,
            # 消息传递的层数
            num_layer=self.mol_num_emb_layer,
            # dropout概率
            dropout=self.dropout,
        )
        # 创建片段嵌入层，用于处理片段级别的信息
        frag_embedding = MPNNEmbedding(
            # 片段节点特征的输入维度
            node_feature_dim=self.frag_node_features,
            # 片段边特征的输入维度
            edge_feature_dim=self.frag_edge_features,
            # 片段节点隐藏层维度
            node_hidden_dim=self.frag_node_hid_dim,
            # 片段边隐藏层维度
            edge_hidden_dim=self.frag_edge_hid_dim,
            # 片段嵌入的层数
            num_layer=self.frag_num_emb_layer,
            # dropout概率
            dropout=self.dropout,
        )
        # 创建片段级消息传递网络，用于片段间的信息交换
        frag_message_passing = MPNNEmbedding(
            # 节点特征维度：分子节点隐藏维度 + 片段节点隐藏维度 + 属性维度*属性数量
            node_feature_dim=self.mol_node_hid_dim
            + self.frag_node_hid_dim
            + self.properties_dim * len(self.properties),
            # 边特征维度：分子节点隐藏维度的2倍（用于表示两个连接节点）
            edge_feature_dim=self.mol_node_hid_dim * 2,
            # 节点隐藏层维度
            node_hidden_dim=self.mol_node_hid_dim,
            # 边隐藏层维度
            edge_hidden_dim=self.mol_node_hid_dim,
            # 片段消息传递的层数
            num_layer=self.frag_message_passing_num_layer,
            # dropout概率
            dropout=self.dropout,
        )

        # 初始化AMPN（原子-分子对消息传递网络）
        self.ampn = AMPN(self.mol_node_features, atom_embedding, frag_embedding)
        # 初始化FMPN（片段级消息传递网络）
        self.fmpn = FMPN(frag_message_passing)
        # 保存片段嵌入层的引用
        self.frag_embedding = frag_embedding

        # 第2部分：确定离开片段的模块
        # 创建位置评分模型，用于预测分子中哪些位置适合进行修改
        self.position_scoring_model = FeedForward(
            # 输入维度：分子节点隐藏维度
            in_dim=self.mol_node_hid_dim,
            # 输出维度：1（单一评分值）
            out_dim=1,
            # 隐藏层维度列表，重复position_score_hid_dim
            hidden_dims=[
                self.position_score_hid_dim
                for _ in range(self.num_mod_position_score_layer)
            ],
            # dropout概率
            dropout=self.dropout,
        )

        # 第3部分：确定插入片段的模块
        # 创建片段评分模型，用于评估候选片段的适合程度
        self.frag_scoring_model = FeedForward(
            # 输入维度：分子节点隐藏维度 + 片段节点隐藏维度
            in_dim=self.mol_node_hid_dim + self.frag_node_hid_dim,
            # 输出维度：1（单一评分值）
            out_dim=1,
            # 隐藏层维度列表
            hidden_dims=[
                self.frag_score_hid_dim for _ in range(self.num_frag_score_layer)
            ],
            # dropout概率
            dropout=self.dropout,
        )

        # 第4部分：附着预测模块
        # 创建附着位点评分模型，用于预测片段与分子的最佳连接位点
        self.attach_scoring_model = FeedForward(
            # 输入维度：分子节点隐藏维度*2 + 片段节点隐藏维度
            in_dim=self.mol_node_hid_dim * 2 + self.frag_node_hid_dim,
            # 输出维度：1（单一评分值）
            out_dim=1,
            # 隐藏层维度列表
            hidden_dims=[
                self.attach_score_hid_dim for _ in range(self.num_attach_score_layer)
            ],
            # dropout概率
            dropout=self.dropout,
        )

        # 第5部分：条件嵌入（已注释掉的代码）
        # 以下代码用于创建基于属性的条件嵌入层
        # self.condition_embeddings = nn.ModuleDict(
        #     {
        #         prop: nn.Linear(
        #             in_features=self.frag_node_hid_dim,
        #             out_features=self.frag_node_hid_dim,
        #         )
        #         for prop in self.properties
        #     }
        # )

        # 创建Sigmoid激活函数，用于将输出压缩到0-1区间
        self.Sigmoid = nn.Sigmoid()
        # 注释掉的二元交叉熵损失函数
        # self.BCELoss = nn.BCELoss(reduction="mean")

        # 初始化模型参数
        self.init_params()

    # 初始化默认设置的私有方法
    def _init_default_setting(self):
        # 获取默认参数字典的所有键，存储为参数列表
        self._PARAMS = list(self.default_args.keys())
        # 将默认参数字典的所有键值对更新到当前实例的属性中
        self.__dict__.update(self.default_args)
        # 返回None（可省略，但保持原代码结构）
        return

    # 初始化模型参数的方法
    def init_params(self):
        # 遍历模型的所有参数
        for param in self.parameters():
            # 如果参数是一维的（如偏置项），跳过初始化
            if param.dim() == 1:
                continue
            # 对于多维参数（如权重矩阵），使用Xavier正态分布初始化
            else:
                nn.init.xavier_normal_(param)

    # 片段嵌入方法，将片段库数据转换为嵌入表示
    def frags_embedding(self, frags_lib_data: Batch) -> Tuple[Tensor, Tensor]:
        """
        返回值说明:
          frags_node_emb: FloatTensor类型，形状为(N, F)，表示片段节点级别的嵌入
          frags_graph_emb: FloatTensor类型，形状为(B, F)，表示片段图级别的嵌入
        """
        # 使用片段嵌入层处理片段的节点特征、边索引和边属性
        frags_node_emb = self.frag_embedding(
            frags_lib_data.x_n, frags_lib_data.edge_index_n, frags_lib_data.edge_attr_n
        )
        # 使用scatter_sum聚合操作，按批次索引将节点嵌入聚合为图级别嵌入
        frags_graph_emb = scatter_sum(
            src=frags_node_emb, index=frags_lib_data.x_n_batch, dim=0
        )

        # 返回节点级别和图级别的片段嵌入
        return frags_node_emb, frags_graph_emb

    # 模型的前向传播方法，这是训练阶段的主要函数
    def forward(
        # 注释掉的旧参数签名
        # self, data: Batch, pos_frags: Batch, neg_frags: Union[Batch, torch.LongTensor]
        # 新的参数：接收批次数据
        self,
        # 批次数据，包含所有训练所需的数据
        batch_data: Batch,
        # 返回类型：六个浮点数值，包括各种损失和概率
    ) -> Tuple[float, float, float, float, float, float]:
        """训练阶段的主要函数。

        参数说明:
          data: .data.PairData对象（形状适用于批次数据）。
            x_n              (bool) : [N,F_n], 节点（原子）特征
            edge_index1_n    (int)  : [2,E_n], 节点（原子）边索引，用于模型1
            edge_index2_n    (int)  : [2,E_n], 节点（原子）边索引，用于模型2
            edge_attr_n      (bool) : [E_n,F_e], 节点（原子）边属性
            x_f              (int)  : [N], 节点到片段的索引映射
            edge_index_f     (int)  : [2,E_f], 片段级连接信息
            edge_attr_f      (int)  : [2,E_f], 每个片段级边的原子索引
            x_n_batch        (int)  : [N], x_n的批次索引
            x_f_batch        (int)  : [F], x_f的批次索引

            smiles           (str)  : [B], 原始分子的SMILES字符串
            y_frag_bool      (bool) : [F], 离开片段的索引
            y_fragID         (int)  : [B], 将要插入的片段ID（答案片段ID）
            allowed_subgraph (str)  : [B], 原始分子允许但为负样本的子图
          pos_frags: torch_geometric.data.Batch对象。
          neg_frags: torch_geometric.data.Batch对象（用于训练），或torch.LongTensor（用于验证）。

        返回值:
          四个损失值和两个概率值。
        """

        # 从批次数据中读取各个组件
        # 提取主要数据
        data = batch_data["data"]
        # 提取正样本片段
        pos_frags = batch_data["pos"]
        # 提取负样本片段
        neg_frags = batch_data["neg"]

        # 第1步：使用AMPN和FMPN对给定分子进行嵌入
        # 通过AMPN（原子-分子对消息传递网络）获得初始嵌入
        ampn_emb = self.ampn(data)
        # 注意：实现选择：条件嵌入向量被添加到AMPN嵌入向量中
        # 如果启用条件嵌入
        if self.conditioning:
            # 初始化条件嵌入列表
            cond_embeddings = []
            # 注释掉的代码：使用条件嵌入层
            # for prop, embedding_layer in self.condition_embeddings.items():
            #     cond_embeddings.append(embedding_layer(batch_data[prop]))
            # 遍历所有属性，直接添加属性值
            for prop in self.properties:
                cond_embeddings.append(batch_data[prop])
            # 将所有条件嵌入沿第二维拼接，形状为[F, num_props]
            condition_embedding = torch.cat(cond_embeddings, dim=1)  # [F, num_props]
            # 将条件嵌入与片段特征拼接，形状为[F, F_node+F_frag+num_props]
            ampn_emb.x_f = torch.cat(
                [ampn_emb.x_f, condition_embedding], dim=1
            )  # [F, F_node+F_frag+num_props]
        # 通过FMPN（片段级消息传递网络）获得最终分子嵌入
        mol_emb = self.fmpn(ampn_emb)

        # 第2步：对离开位置进行评分
        # 对正样本修改位置进行评分
        pos_mod_scores, removal_subgraph_vector = self.mod_pos_scoring(
            mol_emb, data.y_pos_subgraph, data.y_pos_subgraph_idx
        )
        # 对负样本修改位置进行评分
        neg_mod_scores, _ = self.mod_pos_scoring(
            mol_emb, data.y_neg_subgraph, data.y_neg_subgraph_idx
        )

        # 计算正样本位置损失：使用负对数似然损失
        pPosLoss = (pos_mod_scores + 1e-10).log().mean().neg()
        # 计算负样本位置损失：使用scatter_mean按数据点进行平均
        nPosLoss = scatter_mean(
            src=(1 - neg_mod_scores + 1e-10).log(), index=data.y_neg_subgraph_batch
        )  # 按数据点计算平均值
        # 对所有数据点的负样本损失求平均并取负值
        nPosLoss = nPosLoss.mean().neg()

        # 第3步：对插入片段进行评分
        # 处理正样本
        # 获取正样本片段的节点和图级别嵌入
        pos_frag_node_emb, pos_frag_graph_emb = self.frags_embedding(pos_frags)
        # 为正样本片段图嵌入增加一个维度，形状变为[1,B,F]
        pos_frag_graph_emb = pos_frag_graph_emb.unsqueeze(0)  # [1,B,F]
        # 计算正样本片段的评分
        pos_frag_scores = self.frags_scoring(
            removal_subgraph_vector, pos_frag_graph_emb
        )  # [1,B]
        # 计算正样本片段损失
        pFragsLoss = (pos_frag_scores + 1e-12).log().mean().neg()

        # 处理负样本
        # 获取负样本片段的节点和图级别嵌入，只需要图级别嵌入
        _, neg_frag_graph_emb = self.frags_embedding(neg_frags)  # [nB,F]
        # 重塑负样本片段嵌入的形状，变为[n,B,F]
        neg_frag_graph_emb = neg_frag_graph_emb.view(  # [n,B,F]
            self.num_neg_sample, len(data.smiles), neg_frag_graph_emb.size(-1)
        )
        # 计算负样本片段的评分
        neg_frag_scores = self.frags_scoring(
            removal_subgraph_vector, neg_frag_graph_emb
        )  # [n,B]
        # 计算负样本片段损失
        nFragsLoss = (1 - neg_frag_scores + 1e-12).log().mean().neg()

        # 第4步：附着位置预测
        # 计算附着位置的评分
        attachment_scores = self.attachment_scoring(
            data, ampn_emb.x_n, pos_frag_node_emb
        )
        # 获取允许附着的位置（布尔值）
        attachment_allowed = mol_emb.compose_allowed_bool
        # 获取不允许附着的位置（布尔值的反）
        attachment_not_allowed = attachment_allowed == False
        # 调试输出（已注释）
        # print(f"attachment_allowed:\n{attachment_scores*attachment_allowed}")
        # print(f"attachment_not_allowed:\n{attachment_scores*attachment_not_allowed}")
        # 计算附着预测损失：对允许和不允许的位置分别计算损失
        attPredLoss = (
            (
                # 允许附着位置的损失：使用实际评分
                attachment_scores * attachment_allowed
                # 不允许附着位置的损失：使用1减去评分
                + ((1 - attachment_scores) * attachment_not_allowed)
                # 添加小的数值防止log(0)
                + 1e-12
            )
            # 取对数并求平均，然后取负值
            .log()
            .mean()
            .neg()
        )

        # 返回所有损失和平均评分
        return (
            # 正样本位置损失
            pPosLoss,
            # 负样本位置损失
            nPosLoss,
            # 正样本片段损失
            pFragsLoss,
            # 负样本片段损失
            nFragsLoss,
            # 附着预测损失
            attPredLoss,
            # 正样本修改位置的平均评分
            pos_mod_scores.mean(),
            # 负样本修改位置的平均评分
            neg_mod_scores.mean(),
            # 正样本片段的平均评分
            pos_frag_scores.mean(),
            # 负样本片段的平均评分
            neg_frag_scores.mean(),
        )

    # 修改位置评分方法，用于评估分子中哪些子图适合进行修改
    def mod_pos_scoring(
        # 输入参数：分子嵌入数据、子图信息、子图索引
        self, data: Batch, subgraph, subgraph_idx
        # 返回类型：张量元组，包含评分和子图向量
    ) -> Tuple[Tensor, Tensor]:
        """
        参数说明:
          y_pos_subgraph       (int) : [pos_F], 原始分子的正样本子图 *
          y_pos_subgraph_idx   (int) : [pos_F], 原始分子正样本子图的散射索引 *
          y_neg_subgraph       (int) : [neg_F], 原始分子允许但为负样本的子图 *
          y_neg_subgraph_idx   (int) : [neg_F], 原始分子允许但为负样本子图的散射索引 *
          y_neg_subgraph_batch (int) : [neg_F], 批次索引
          示例:
            y_neg_subgraph       = tensor([ 0,  1,  2,  0,  1,  4,  5,  6,  4,  5, 16, 17, 18, 16, 17])
            y_neg_subgraph_idx   = tensor([ 0,  1,  2,  3,  3,  4,  5,  6,  7,  7,  8,  9, 10, 11, 11])
            y_neg_subgraph_batch = tensor([ 0,  0,  0,  0,  1,  1,  1,  1,  2,  2,  2,  2])

        返回值:
          pos_frag_level_mod_score : [B] 正样本片段级修改评分
          neg_frag_level_mod_score : [?] 负样本片段级修改评分，每个分子的负片段数量不一定相同
          pos_subgraph_vector : [B,F_n] 正样本子图向量

        """

        # 使用scatter_sum聚合操作，将子图中的片段特征按索引聚合
        subgraph_vector = scatter_sum(
            src=data.x_f[subgraph], index=subgraph_idx, dim=0
        )  # 形状：[B, F_n]
        # 通过位置评分模型计算片段级修改评分，并压缩最后一个维度
        frag_level_mod_score = self.position_scoring_model(subgraph_vector).squeeze(
            -1
        )  # 形状：[B]
        # 使用Sigmoid函数将评分转换为概率值
        frag_level_mod_prob = self.Sigmoid(frag_level_mod_score)

        # 返回修改概率和子图向量
        return frag_level_mod_prob, subgraph_vector

    # 片段评分方法，用于评估插入片段的适合程度
    def frags_scoring(self, removal_subgraph_vector: Tensor, frag_h: Tensor) -> Tensor:
        """
        参数说明:
          removal_subgraph_vector:
            通过AMPN和FMPN嵌入的片段级隐藏向量。
            形状: [B,F_n]。B: 批次数量
          frag_h:
            通过self.frag_embedding嵌入的片段隐藏向量。
            形状: [n,B,F_f]。n: 样本数量, B: 批次数量
              对于正样本，n = 1
              对于负样本，n = n_sample
        返回值:
          graph_wise_frags_score: [n,B] 图级别的片段评分
        """

        # 获取样本数量（第一个维度）
        n_sample = frag_h.size(0)
        # 将移除子图向量扩展到与片段向量相同的样本数量
        removal_subgraph_vector = removal_subgraph_vector.unsqueeze(0).repeat(
            n_sample, 1, 1
        )  # 形状：[n,B,F_n]
        # 将移除子图向量和片段向量在最后一个维度拼接
        concat_query = torch.cat(
            [removal_subgraph_vector, frag_h], dim=2
        )  # 形状：[n,B,F_n+F_f]
        # 通过片段评分模型计算评分，并压缩最后一个维度
        graph_wise_frags_score = self.frag_scoring_model(concat_query).squeeze(-1)
        # 使用Sigmoid函数将评分转换为概率值
        graph_wise_frags_score = self.Sigmoid(graph_wise_frags_score)  # 形状：[n,B]

        # 返回图级别的片段评分
        return graph_wise_frags_score

    # 附着评分方法，用于预测片段与分子的最佳连接位点
    def attachment_scoring(
        # 输入参数：数据、分子节点嵌入、片段节点嵌入
        self, data: PairData, mol_node_emb: Tensor, frag_node_emb: Tensor
        # 返回张量类型
    ) -> Tensor:
        """
        参数说明:
          data:
            包含compose_original_idx, compose_fragment_idx, compose_allowed_bool
            形状: [B]。B: 批次数量
          mol_node_emb:
            通过AMPN嵌入的分子。
            形状: [N_n, 2*F_n]。N_n: 原始分子中的原子数量
          frag_node_emb:
            通过frags_embedding嵌入的片段。
            形状: [N_f, F_f]。N_f: 新片段中的原子数量

        返回值:
          attachment_scores: [n,B] 附着评分
        """

        # 根据索引获取原始分子中相应原子的嵌入
        atom_emb_in_original = mol_node_emb[data.compose_original_idx]
        # 根据索引获取新片段中相应原子的嵌入
        atom_emb_in_new_frag = frag_node_emb[data.compose_fragment_idx]

        # 将原始分子原子嵌入和新片段原子嵌入在第二个维度拼接
        concat_query = torch.cat([atom_emb_in_original, atom_emb_in_new_frag], dim=1)
        # 通过附着评分模型计算评分，并压缩最后一个维度
        attachment_scores = self.attach_scoring_model(concat_query).squeeze(-1)
        # 使用Sigmoid函数将评分转换为概率值
        attachment_scores = self.Sigmoid(attachment_scores)  # 形状：[num_combinations]

        # 返回附着评分
        return attachment_scores

    # 静态方法：解析SMILES字符串为数据对象
    @staticmethod
    def parse_smi(smi: str, type="Mol") -> Data:
        # 调用from_smiles函数将SMILES字符串转换为数据对象
        return from_smiles(smi, type=type)

    # 静态方法：解析分子对象为数据对象
    @staticmethod
    def parse_mol(mol: Mol, type="Mol") -> Data:
        # 调用from_mol函数将分子对象转换为数据对象
        return from_mol(mol, type=type)

    # 类方法：从训练好的模型文件创建模型实例
    @classmethod
    def from_trained_model(
        # 类对象
        cls,
        # 模型文件的路径
        path_to_model: Union[str, pathlib.Path],
        # 属性列表，默认为空
        properties: List[PROPERTY] = [],
        # 其他关键字参数
        **kwargs,
    ):
        # 如果指定了属性，将属性名转换为小写
        if len(properties) > 0:
            properties = [prop.lower() for prop in properties]

        # 创建参数解析器对象
        model_args = argparse.ArgumentParser()
        # 如果有额外的关键字参数，添加到模型参数中
        if kwargs:
            for k, v in kwargs.items():
                vars(model_args)[k] = v

        # 设置属性参数
        vars(model_args)["properties"] = properties
        # 创建模型实例
        model = cls(model_args)
        # 恢复模型权重
        model.restore(path_to_model)
        # 返回模型实例
        return model

    # 静态方法：保存模型权重
    @staticmethod
    def save_model(model_state_dict, save_dir, name="Best_model"):
        # 使用torch.save保存模型状态字典到指定路径
        torch.save(model_state_dict, f"{save_dir}/{name}.pt")

    # 恢复模型权重的方法
    def restore(self, path_to_model: Union[str, pathlib.Path]):
        # 将路径转换为字符串格式
        path_to_model = str(path_to_model)
        # 如果设备是CPU，指定map_location为CPU
        if self.device == torch.device("cpu"):
            model_params = torch.load(path_to_model, map_location=torch.device("cpu"))
        # 否则正常加载
        else:
            model_params = torch.load(path_to_model)
        # 加载模型状态字典
        self.load_state_dict(model_params)
        # 保存模型路径
        self.path_to_model = path_to_model

    # 将模型移动到CUDA设备的方法
    def cuda(self, device: str = "cuda:0"):
        # 调用父类的cuda方法
        super().cuda(device)
        # 更新设备信息，从片段评分模型的权重设备获取
        self.device = self.frag_scoring_model.fcs[0].weight.device
        # 返回self以支持链式调用
        return self

    # 将模型移动到指定设备的方法
    def to(self, torch_device):
        # 调用父类的to方法
        super().to(torch_device)
        # 更新设备信息，从片段评分模型的权重设备获取
        self.device = self.frag_scoring_model.fcs[0].weight.device
        # 返回self以支持链式调用
        return self

    # 模型的字符串表示方法
    def __repr__(self):
        # 构建模型表示字符串，从类名开始
        model_repr = f"{self.__class__.__name__}(\n"
        # 遍历实例字典中的所有属性
        for arg, value in self.__dict__.items():
            # 只显示在参数列表中的属性
            if arg in self._PARAMS:
                model_repr += f"  {arg}: {value}\n"
        # 添加结束括号
        model_repr += ")\n"
        # 添加模块信息
        model_repr += str(self._modules)
        # 返回完整的字符串表示
        return model_repr
