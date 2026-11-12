import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter_add


# =====================================================================
#  通道一:标准图注意力  (组合:part -> whole)
# =====================================================================
class PartToWholeConv(MessagePassing):
    """汽车聚合它的零件:经典 GAT 风格的注意力聚合。"""

    def __init__(self, in_dim, out_dim, heads=1, dropout=0.0):
        super().__init__(aggr='add', node_dim=0)
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = dropout

        # 源、目标各自投影(异质图维度可能不同)
        self.lin_src = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.lin_dst = nn.Linear(in_dim, heads * out_dim, bias=False)
        # 注意力打分参数:[1, heads, 2*out_dim]
        self.att = nn.Parameter(torch.empty(1, heads, 2 * out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x, edge_index):
        # x = (x_src, x_dst);异质二部传播
        H, C = self.heads, self.out_dim
        x_src = self.lin_src(x[0]).view(-1, H, C)
        x_dst = self.lin_dst(x[1]).view(-1, H, C)
        out = self.propagate(edge_index, x=(x_src, x_dst))
        return out.mean(dim=1)  # 多头平均,输出 [N_dst, out_dim]

    def message(self, x_i, x_j, index, ptr, size_i):
        # x_i: 目标(whole), x_j: 源(part),形状 [E, H, C]
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)        # 按目标节点归一化
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)                  # [E, H, C]


# =====================================================================
#  通道二:共性方向  (抽象:whole -> part)
# =====================================================================
class WholeToPartConv(MessagePassing):
    """轮子 = 汽车/摩托车/三轮车的共性主方向(幂迭代近似第一奇异向量)。"""

    def __init__(self, in_dim, out_dim, power_iters=5):
        super().__init__(aggr=None, node_dim=0)   # 自定义聚合
        self.out_dim = out_dim
        self.power_iters = power_iters
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index, num_dst):
        x_src = self.lin(x[0])   # whole 邻居(投影后)
        # 目标节点数显式传入,避免孤立节点导致 dim_size 推断错误
        return self.propagate(edge_index, x=(x_src, None),
                              dim_size=num_dst, num_dst=num_dst)

    def message(self, x_j):
        return x_j   # 暂存,真正归约在 aggregate

    def aggregate(self, inputs, index, num_dst, dim_size=None):
        return self._common_direction(inputs, index, num_dst)

    def _common_direction(self, x, index, dim_size):
        """逐 part 节点,用幂迭代求其 whole 邻居的第一主方向并投影。"""
        d = x.size(-1)
        device = x.device

        # 1) 初始化方向向量(不参与梯度,纯作迭代起点)
        v = torch.randn(dim_size, d, device=device)
        v = F.normalize(v, dim=-1, eps=1e-8).detach()

        # 2) 幂迭代:v <- (XᵀX)v / ||(XᵀX)v||,完全向量化
        for _ in range(self.power_iters):
            coeff = (x * v[index]).sum(dim=-1, keepdim=True)        # x_jᵀv
            Mv = scatter_add(coeff * x, index, dim=0, dim_size=dim_size)
            v = F.normalize(Mv, dim=-1, eps=1e-8)

        # 3) 符号对齐:与邻居均值同向,消除 ±v 的不确定性
        mean = scatter_add(x, index, dim=0, dim_size=dim_size)
        sign = torch.sign((v * mean).sum(dim=-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        v = sign * v

        # 4) 投影回去,保留语义强度
        proj = (x * v[index]).sum(dim=-1, keepdim=True)            # x_jᵀv
        strength = scatter_add(proj, index, dim=0, dim_size=dim_size)
        cnt = scatter_add(torch.ones_like(proj), index,
                          dim=0, dim_size=dim_size)
        out = (strength / (cnt + 1e-8)) * v

        # 孤立节点(无邻居)兜底:输出置零,避免 NaN 污染
        mask = (cnt.squeeze(-1) > 0).float().unsqueeze(-1)
        return out * mask


# =====================================================================
#  主模块:双通道图注意力 + 可学习 λ 门控
# =====================================================================
class DualChannelGAT(nn.Module):
    """
    一个统一的双通道图注意力层:
      - 通道一(组合): part -> whole,标准 GAT
      - 通道二(抽象): whole -> part,共性主方向
      - λ 可学习门控,把两通道在 part 节点上融合
    """

    def __init__(self, in_dims, out_dim, heads=1,
                 power_iters=5, adaptive=True, dropout=0.0,
                 use_ch1=True, use_ch2=True):
        super().__init__()
        self.adaptive = adaptive
        self.use_ch1 = use_ch1
        self.use_ch2 = use_ch2

        # part 侧自身投影 (始终需要: Ch1-only 时 part 侧仅靠它)
        self.lin_part_self = nn.Linear(in_dims['part'], out_dim, bias=False)
        nn.init.xavier_uniform_(self.lin_part_self.weight)

        # 通道一: part -> whole GAT (禁用时 whole 侧退化为自身投影)
        if use_ch1:
            self.conv_p2w = PartToWholeConv(in_dims['part'], out_dim,
                                            heads=heads, dropout=dropout)
        else:
            self.lin_whole_self = nn.Linear(in_dims['whole'], out_dim, bias=False)
            nn.init.xavier_uniform_(self.lin_whole_self.weight)

        # 通道二: whole -> part 共性方向 (禁用时 part 侧仅用自身投影, λ≡1)
        if use_ch2:
            self.conv_w2p = WholeToPartConv(in_dims['whole'], out_dim,
                                            power_iters=power_iters)
            # 可学习 λ:全局基准 logit,初始化为 0 => sigmoid=0.5 等权
            self.lam_logit = nn.Parameter(torch.zeros(1))
            if adaptive:
                self.gate = nn.Linear(2 * out_dim, 1)
                nn.init.zeros_(self.gate.weight)
                nn.init.zeros_(self.gate.bias)

    def forward(self, x_dict, edge_index_dict):
        x_part = x_dict['part']
        x_whole = x_dict['whole']
        num_part = x_part.size(0)

        # ---- 通道一:更新 whole 节点 ----
        if self.use_ch1:
            m_whole = self.conv_p2w(
                (x_part, x_whole),
                edge_index_dict[('part', 'composes', 'whole')])
        else:
            m_whole = self.lin_whole_self(x_whole)

        # ---- 通道二:更新 part 节点(共性方向)+ λ 门控 ----
        if self.use_ch2:
            m_part_common = self.conv_w2p(
                (x_whole, None),
                edge_index_dict[('whole', 'abstracts', 'part')],
                num_dst=num_part)
            m_part_self = self.lin_part_self(x_part)
            if self.adaptive:
                gate_in = torch.cat([m_part_self, m_part_common], dim=-1)
                lam = torch.sigmoid(self.lam_logit + self.gate(gate_in))
            else:
                lam = torch.sigmoid(self.lam_logit)
            m_part = lam * m_part_self + (1.0 - lam) * m_part_common
        else:
            m_part = self.lin_part_self(x_part)

        # 诊断用:存储本层中间量 (β 融合前两通道输出与 λ)
        self._lam = lam if self.use_ch2 else torch.ones(num_part, 1, device=x_part.device)
        self._m_part = m_part
        self._m_whole = m_whole
        return {'whole': m_whole, 'part': m_part}

    @property
    def lambda_value(self):
        """方便训练时监控全局 λ 基准的取值 (Ch2 禁用时恒为 1)。"""
        if not self.use_ch2:
            return 1.0
        return torch.sigmoid(self.lam_logit).item()
