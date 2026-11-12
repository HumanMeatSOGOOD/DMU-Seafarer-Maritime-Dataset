"""
=============================================================================
全边联合训练 v2 — 支持多种编码器, 在所有 19 种边上联合训练
=============================================================================

用法:
  python all_edge_train.py --encoder graphsage
  python all_edge_train.py --encoder gat
  ...  (graphconv, graphsage, gat, film, gin, han, gatv2)

输出:
  数据/节点嵌入/alledge_{encoder}_embeddings.pt
  数据/节点嵌入/alledge_{encoder}_decoder.pt
  数据/alledge_link_results.json                  # 所有方法的逐边 AUC
=============================================================================
"""

import sys, os, math, json, time, argparse, copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    HeteroConv, SAGEConv, GraphConv, GATConv, FiLMConv, GINConv,
    GATv2Conv, HANConv,
)
import warnings
warnings.filterwarnings('ignore')

# DualChannelGAT 在同目录下
sys.path.insert(0, os.path.dirname(__file__))
from DualChannelGAT import DualChannelGAT

HIDDEN_DIM = 128; OUT_DIM = 64; NUM_LAYERS = 2
EPOCHS = 300; LR = 0.001; HEADS = 4
MSG_RATIO = 0.7; VAL_RATIO = 0.1; TEST_RATIO = 0.1
PATIENCE = 40; SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '数据')
GRAPH_PATH = os.path.join(DATA_DIR, 'hetero_graph_directed.pt')
EMBED_DIR = os.path.join(DATA_DIR, '节点嵌入')
RESULTS_PATH = os.path.join(DATA_DIR, 'alledge_link_results.json')

# ============================================================
# 1. 编码器工厂
# ============================================================

def build_encoder(enc_type, data):
    init_embeds = nn.ModuleDict({
        nt: nn.Embedding(data[nt].num_nodes, HIDDEN_DIM) for nt in data.node_types
    })
    convs = nn.ModuleList()
    proj = nn.ModuleDict({
        nt: nn.Linear(HIDDEN_DIM, OUT_DIM) for nt in data.node_types
    })

    # GAT/GATv2/HAN 在二分边上的卷积没有自环: 当某类关系里节点只有 1 个邻居时,
    # softmax 权重恒为 1, 输出与节点自身无关 (实测 crew 只剩 5,102 个不同嵌入)。
    # 给这三类模型补一个残差自身投影, 与 GraphSAGE/GraphConv/GIN/FiLM 的 self 项对齐。
    self_proj = None
    if enc_type in ('gat', 'gatv2', 'han'):
        self_proj = nn.ModuleDict({
            nt: nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for nt in data.node_types
        })

    if enc_type == 'graphconv':
        for _ in range(NUM_LAYERS):
            d = {et: GraphConv(HIDDEN_DIM, HIDDEN_DIM) for et in data.edge_types}
            convs.append(HeteroConv(d, aggr='mean'))

    elif enc_type == 'graphsage':
        for _ in range(NUM_LAYERS):
            d = {et: SAGEConv(HIDDEN_DIM, HIDDEN_DIM, root_weight=True) for et in data.edge_types}
            convs.append(HeteroConv(d, aggr='mean'))

    elif enc_type == 'gat':
        for _ in range(NUM_LAYERS):
            d = {et: GATConv(HIDDEN_DIM, HIDDEN_DIM // HEADS, heads=HEADS,
                             dropout=0.0, add_self_loops=False) for et in data.edge_types}
            convs.append(HeteroConv(d, aggr='mean'))

    elif enc_type == 'film':
        for _ in range(NUM_LAYERS):
            d = {et: FiLMConv(HIDDEN_DIM, HIDDEN_DIM) for et in data.edge_types}
            convs.append(HeteroConv(d, aggr='mean'))

    elif enc_type == 'gin':
        for _ in range(NUM_LAYERS):
            d = {}
            for et in data.edge_types:
                mlp = nn.Sequential(nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
                                    nn.Linear(HIDDEN_DIM, HIDDEN_DIM))
                d[et] = GINConv(mlp, train_eps=True)
            convs.append(HeteroConv(d, aggr='sum'))

    elif enc_type == 'han':
        for _ in range(NUM_LAYERS):
            convs.append(HANConv(
                in_channels={nt: HIDDEN_DIM for nt in data.node_types},
                out_channels=HIDDEN_DIM, metadata=data.metadata(), heads=HEADS, dropout=0.0,
            ))

    elif enc_type == 'gatv2':
        for _ in range(NUM_LAYERS):
            d = {et: GATv2Conv(HIDDEN_DIM, HIDDEN_DIM // HEADS, heads=HEADS,
                               dropout=0.0, add_self_loops=False) for et in data.edge_types}
            convs.append(HeteroConv(d, aggr='mean'))

    elif enc_type == 'dualchannel':
        # 旧方案: 每种边一个独立 DC 实例
        return HeteroDualChannelEncoder(data, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                                         num_layers=NUM_LAYERS)

    elif enc_type == 'unified_dc':
        # 新方案: 所有边合并到 1 个 DC
        return UnifiedDualChannelEncoder(data, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                                          num_layers=NUM_LAYERS)
    elif enc_type == 'unified_dc_ch1only':
        # 消融: 仅通道一 (禁用 Ch2 共性方向)
        return UnifiedDualChannelEncoder(data, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                                          num_layers=NUM_LAYERS, use_ch2=False)
    elif enc_type == 'unified_dc_ch2only':
        # 消融: 仅通道二 (禁用 Ch1 GAT)
        return UnifiedDualChannelEncoder(data, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                                          num_layers=NUM_LAYERS, use_ch1=False)
    elif enc_type == 'unified_dc_concat':
        # 稳定性修复: concat+投影融合 (替代加性 β 融合, 避免相消)
        return UnifiedDualChannelEncoder(data, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                                          num_layers=NUM_LAYERS, fusion='concat')

    else:
        raise ValueError(f"Unknown encoder: {enc_type}")

    return Encoder(init_embeds, convs, proj, enc_type, self_proj)


class Encoder(nn.Module):
    def __init__(self, init_embeds, convs, proj, enc_type, self_proj=None):
        super().__init__()
        self.init_embeds = init_embeds
        self.convs = convs
        self.proj = proj
        self.enc_type = enc_type
        self.self_proj = self_proj

    def forward(self, edge_index_dict):
        x = {nt: self.init_embeds[nt].weight for nt in self.init_embeds}

        for conv in self.convs:
            new_x = conv(x, edge_index_dict)
            for k in x:
                sp = self.self_proj[k](x[k]) if self.self_proj is not None else None
                if k not in new_x or new_x[k] is None:
                    new_x[k] = sp if sp is not None else x[k]
                elif sp is not None:
                    new_x[k] = new_x[k] + sp
            x = {k: F.relu(v) for k, v in new_x.items()}

        return {k: self.proj[k](v) for k, v in x.items()}


# ============================================================
# 1b. 统一双通道编码器 (UnifiedDualChannel — 1 个 DC 处理全部 19 边)
# ============================================================

class UnifiedDualChannelEncoder(nn.Module):
    """
    将所有 19 种边合并为一张大二分图, 用单个 DualChannelGAT 实例处理。

    与旧的 19-DC 方案的关键区别:
      - 每种节点类型有两个投影矩阵: P_part (作为下级时) 和 P_whole (作为上级时)
      - 所有边索引加类型偏移, 合并为 (part_global_idx, whole_global_idx)
      - 1 个 DC 处理全部 155K 条边, 参数共享, 小边受益于大边学到的模式
      - DC 的输出 (part, whole) 取均值, 按偏移拆分回各节点类型
    """

    def __init__(self, data, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM,
                 num_layers=NUM_LAYERS, power_iters=3,
                 use_ch1=True, use_ch2=True, fusion='beta'):
        super().__init__()
        self.node_types = sorted(data.node_types)
        self.edge_types = sorted(data.edge_types)
        self.num_layers = num_layers
        self.use_ch1 = use_ch1
        self.use_ch2 = use_ch2
        self.fusion = fusion

        # 初始嵌入
        self.init_embeds = nn.ModuleDict({
            nt: nn.Embedding(data[nt].num_nodes, hidden_dim) for nt in self.node_types
        })

        # 每种节点类型的双身份投影 (单层 Linear, 已验证最有效)
        self.P_part = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types
        })
        self.P_whole = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, hidden_dim) for nt in self.node_types
        })


        # 计算全局偏移
        self.offsets = {}
        offset = 0
        for nt in self.node_types:
            self.offsets[nt] = offset
            offset += data[nt].num_nodes
        self.total_nodes = offset

        self.node_slices = {}
        for nt in self.node_types:
            self.node_slices[nt] = (self.offsets[nt],
                                     self.offsets[nt] + data[nt].num_nodes)

        # 1 个共享 DC (可堆叠多层)
        self.dc_layers = nn.ModuleList([
            DualChannelGAT(
                in_dims={'part': hidden_dim, 'whole': hidden_dim},
                out_dim=hidden_dim, heads=1, power_iters=power_iters,
                adaptive=True, dropout=0.0,
                use_ch1=use_ch1, use_ch2=use_ch2,
            ) for _ in range(num_layers)
        ])

        # 可学习的全局融合系数 β = σ(logit), 初始 0.5 等权 (fusion='beta' 时使用)
        self.beta_logit = nn.Parameter(torch.zeros(1))

        # concat 融合: 拼接两通道后投影回 hidden_dim (避免加性相消, fusion='concat' 时使用)
        if fusion == 'concat':
            self.fusion_proj = nn.ModuleList([
                nn.Linear(2 * hidden_dim, hidden_dim) for _ in range(num_layers)
            ])

        # 输出投影
        self.project = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, out_dim) for nt in self.node_types
        })

        total_p = sum(p.numel() for p in self.parameters())
        print(f"  [UnifiedDC] 参数量: {total_p:,}")

    def _merge_edges(self, edge_index_dict):
        """把传入的（msg split）边按节点偏移合并成全局边表。"""
        parts, wholes = [], []
        for et in self.edge_types:
            ei = edge_index_dict.get(et)
            if ei is None or ei.numel() == 0:
                continue
            pt, rel, wt = et
            parts.append(ei[0] + self.offsets[pt])
            wholes.append(ei[1] + self.offsets[wt])
        return torch.stack([torch.cat(parts), torch.cat(wholes)], dim=0)

    def forward(self, edge_index_dict):
        """只用传入的边做消息传递（训练/评测都传 70% msg split）。"""
        x = {nt: self.init_embeds[nt].weight for nt in self.node_types}  # 128-dim

        merged = self._merge_edges(edge_index_dict)

        for l_idx, dc_layer in enumerate(self.dc_layers):
            # ---- 构建全局 part / whole 向量 ----
            part_vecs, whole_vecs = [], []
            for nt in self.node_types:
                part_vecs.append(self.P_part[nt](x[nt]))
                whole_vecs.append(self.P_whole[nt](x[nt]))
            x_part = torch.cat(part_vecs, dim=0)
            x_whole = torch.cat(whole_vecs, dim=0)

            # ---- 单 DC 处理全图 ----
            out = dc_layer(
                {'part': x_part, 'whole': x_whole},
                {
                    ('part', 'composes', 'whole'): merged,
                    ('whole', 'abstracts', 'part'): merged.flip(0),
                },
            )

            # ---- 通道融合 ----
            if self.fusion == 'concat':
                merged_vec = self.fusion_proj[l_idx](
                    torch.cat([out['part'], out['whole']], dim=-1))
                for nt in self.node_types:
                    s, e = self.node_slices[nt]
                    x[nt] = merged_vec[s:e]
            else:  # beta 加权融合
                beta = torch.sigmoid(self.beta_logit)
                for nt in self.node_types:
                    s, e = self.node_slices[nt]
                    x[nt] = beta * out['part'][s:e] + (1.0 - beta) * out['whole'][s:e]

        # 最终投影到 OUT_DIM
        return {nt: self.project[nt](x[nt]) for nt in self.node_types}


# ============================================================
# 1c. 异构双通道编码器 (19 DC — 旧方案, 保留用于对比)
# ============================================================

class HeteroDualChannelEncoder(nn.Module):
    """每种边类型一个独立的 DualChannelGAT 实例。"""

    def __init__(self, data, hidden_dim=HIDDEN_DIM, out_dim=OUT_DIM, num_layers=NUM_LAYERS):
        super().__init__()
        self.node_types = sorted(data.node_types)
        self.edge_types = sorted(data.edge_types)

        self.init_embeds = nn.ModuleDict({
            nt: nn.Embedding(data[nt].num_nodes, hidden_dim) for nt in self.node_types
        })

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict()
            for et in self.edge_types:
                pt, rel, wt = et
                key = f"{pt}||{rel}||{wt}"
                layer[key] = DualChannelGAT(
                    in_dims={'part': hidden_dim, 'whole': hidden_dim},
                    out_dim=hidden_dim, heads=1, power_iters=3, adaptive=True, dropout=0.0,
                )
            self.layers.append(layer)

        self.project = nn.ModuleDict({
            nt: nn.Linear(hidden_dim, out_dim) for nt in self.node_types
        })

    def forward(self, edge_index_dict):
        x = {nt: self.init_embeds[nt].weight for nt in self.node_types}

        for layer in self.layers:
            updates = {nt: [] for nt in self.node_types}

            for edge_key, dc_gat in layer.items():
                pt, rel, wt = edge_key.split('||', 2)
                directed_edge = edge_index_dict.get((pt, rel, wt))
                if directed_edge is None or directed_edge.shape[1] == 0:
                    continue

                dc_edge_dict = {
                    ('part', 'composes', 'whole'): directed_edge,
                    ('whole', 'abstracts', 'part'): directed_edge.flip(0),
                }
                out = dc_gat({'part': x[pt], 'whole': x[wt]}, dc_edge_dict)
                updates[pt].append(out['part'])
                updates[wt].append(out['whole'])

            x_new = {}
            for nt in self.node_types:
                if len(updates[nt]) > 0:
                    x_new[nt] = torch.stack(updates[nt]).mean(dim=0)
                else:
                    x_new[nt] = x[nt]
            x = x_new

        return {nt: self.project[nt](x[nt]) for nt in self.node_types}


# ============================================================
# 2. 解码器 + 工具
# ============================================================

class UniversalDecoder(nn.Module):
    def __init__(self, dim=OUT_DIM):
        super().__init__()
        self.W = nn.Linear(dim, dim)
    def forward(self, e_src, e_dst, src_idx, dst_idx):
        return (self.W(e_src[src_idx]) * e_dst[dst_idx]).sum(dim=-1)


def split_edges(edge_index):
    num = edge_index.shape[1]
    if num < 4:
        return edge_index, edge_index[:, :0], edge_index[:, :0], edge_index[:, :0]
    perm = torch.randperm(num)
    nm = int(num*MSG_RATIO); nv = int(num*VAL_RATIO); nt = int(num*TEST_RATIO)
    ns = num - nm - nv - nt
    return (edge_index[:, perm[:nm]], edge_index[:, perm[nm:nm+ns]],
            edge_index[:, perm[nm+ns:nm+ns+nv]], edge_index[:, perm[nm+ns+nv:]])

def sample_neg(n_src, n_dst, n_pos, pos_keys=None, generator=None):
    """Uniform negative sampling.

    If pos_keys (packed u*n_dst+v of all known positive edges) is given, pairs
    that are real edges are resampled, so a "negative" is never a positive.
    """
    u = torch.randint(0, n_src, (n_pos,), generator=generator)
    v = torch.randint(0, n_dst, (n_pos,), generator=generator)
    if pos_keys is not None and pos_keys.numel() > 0:
        for _ in range(20):
            bad = torch.isin(u * n_dst + v, pos_keys)
            nb = int(bad.sum())
            if nb == 0:
                break
            u[bad] = torch.randint(0, n_src, (nb,), generator=generator)
            v[bad] = torch.randint(0, n_dst, (nb,), generator=generator)
    return torch.stack([u, v], dim=0)

def compute_auc(pos, neg):
    """Tie-aware ROC-AUC.

    Uses sklearn so that equal scores receive 0.5 credit.  The previous
    hand-rolled rank formula used the np(np+1)/2 correction instead of
    np(np-1)/2 and therefore added exactly 1/n_neg to every AUC (it even
    returned values > 1).
    """
    from sklearn.metrics import roc_auc_score
    y = torch.cat([torch.ones(pos.shape[0]), torch.zeros(neg.shape[0])]).numpy()
    s = torch.cat([pos, neg]).detach().cpu().numpy()
    return float(roc_auc_score(y, s))


# ============================================================
# 3. 训练
# ============================================================

def train_one_encoder(enc_type, seed=SEED, save_suffix=''):
    torch.manual_seed(seed)
    print(f"\n{'='*60}")
    print(f"  全边训练: {enc_type.upper()}")
    print(f"{'='*60}")

    data = torch.load(GRAPH_PATH, weights_only=False)
    edge_types = sorted(data.edge_types)
    ntypes = sorted(data.node_types)
    node_counts = {nt: data[nt].num_nodes for nt in ntypes}

    # 划分 (先去掉完全重复的 (u,v), 避免同一对边同时落进 msg 和 test)
    graph_edges = {et: torch.unique(data[et].edge_index, dim=1) for et in edge_types}
    splits = {}
    for et in edge_types:
        splits[et] = dict(zip(['msg', 'train_sup', 'val', 'test'],
                              split_edges(graph_edges[et])))

    # 已知正边的 packed key, 用于负采样时过滤假负样本
    pos_keys = {}
    for et in edge_types:
        src_t, _, dst_t = et
        ei = graph_edges[et]
        pos_keys[et] = torch.sort(ei[0] * node_counts[dst_t] + ei[1]).values

    # val/test 使用固定负样本: 所有方法、所有 epoch 都在同一组负样本上比较
    eval_gen = torch.Generator().manual_seed(100000 + seed)
    val_neg = {et: sample_neg(node_counts[et[0]], node_counts[et[2]],
                              splits[et]['val'].shape[1], pos_keys[et], eval_gen)
               for et in edge_types}
    test_neg = {et: sample_neg(node_counts[et[0]], node_counts[et[2]],
                               splits[et]['test'].shape[1], pos_keys[et], eval_gen)
                for et in edge_types}

    # sqrt 权重
    edge_weights = {}
    for et in edge_types:
        n = max(splits[et]['train_sup'].shape[1], 1)
        edge_weights[et] = math.sqrt(n)
    tw = sum(edge_weights.values())
    for et in edge_weights:
        edge_weights[et] /= tw

    msg_ed = {et: splits[et]['msg'] for et in edge_types}

    model = build_encoder(enc_type, data).to(DEVICE)
    decoder = UniversalDecoder().to(DEVICE)
    n_param = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in decoder.parameters())
    opt = torch.optim.Adam(list(model.parameters())+list(decoder.parameters()), lr=LR)

    msg_ed = {k: v.to(DEVICE) for k, v in msg_ed.items()}
    for et in splits:
        for k in splits[et]:
            splits[et][k] = splits[et][k].to(DEVICE)
    val_neg = {k: v.to(DEVICE) for k, v in val_neg.items()}
    test_neg = {k: v.to(DEVICE) for k, v in test_neg.items()}

    best_avg, best_st, pc = 0.0, None, 0

    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        total_loss = 0.0
        z = model(msg_ed)

        for et in edge_types:
            ts = splits[et]['train_sup']
            if ts.shape[1] == 0: continue
            src_t, _, dst_t = et
            pp = decoder(z[src_t], z[dst_t], ts[0], ts[1])
            ne = sample_neg(node_counts[src_t], node_counts[dst_t], ts.shape[1],
                            pos_keys[et]).to(DEVICE)
            pn = decoder(z[src_t], z[dst_t], ne[0], ne[1])
            total_loss = total_loss + (
                -torch.log(torch.sigmoid(pp)+1e-15).mean()
                -torch.log(1-torch.sigmoid(pn)+1e-15).mean()) * edge_weights[et]

        total_loss.backward(); opt.step()

        if (ep+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                z = model(msg_ed)
                val_aucs = []
                for et in edge_types:
                    ve = splits[et]['val']
                    if ve.shape[1] < 2: continue
                    src_t, _, dst_t = et
                    vp = decoder(z[src_t], z[dst_t], ve[0], ve[1])
                    ne = val_neg[et]
                    vn = decoder(z[src_t], z[dst_t], ne[0], ne[1])
                    val_aucs.append(compute_auc(vp, vn))
                avg_auc = sum(val_aucs)/len(val_aucs) if val_aucs else 0.0
                print(f"  Epoch {ep+1:3d} | Loss: {total_loss.item():.3f} | Avg Val AUC: {avg_auc:.4f}")
                if avg_auc > best_avg:
                    best_avg = avg_auc
                    best_st = {'model': copy.deepcopy(model.state_dict()),
                               'decoder': copy.deepcopy(decoder.state_dict()),
                               'epoch': ep}
                    pc = 0
                else:
                    pc += 1
                    if pc >= PATIENCE // 10:
                        print(f"  早停于 epoch {ep+1}")
                        break

    # Test AUC
    model.load_state_dict(best_st['model']); decoder.load_state_dict(best_st['decoder'])
    model.eval(); decoder.eval()
    with torch.no_grad():
        z = model(msg_ed)
        test_aucs = {}
        for et in edge_types:
            te = splits[et]['test']
            if te.shape[1] < 2: continue
            src_t, _, dst_t = et
            tp = decoder(z[src_t], z[dst_t], te[0], te[1])
            ne = test_neg[et]
            tn = decoder(z[src_t], z[dst_t], ne[0], ne[1])
            test_aucs[f"{src_t} -> {dst_t}"] = compute_auc(tp, tn)
        embeddings = {k: v.cpu() for k, v in z.items()}

    avg_test = sum(test_aucs.values()) / len(test_aucs)
    test_aucs['_avg'] = avg_test
    print(f"  平均 Test AUC: {avg_test:.4f} (best epoch {best_st['epoch']+1})")

    # 保存
    suffix = f'_{save_suffix}' if save_suffix else ''
    torch.save(embeddings, os.path.join(EMBED_DIR, f'alledge_{enc_type}{suffix}_embeddings.pt'))
    if not save_suffix:  # 仅主运行保存 decoder (推荐评测用余弦, 不需 decoder)
        torch.save(decoder.state_dict(), os.path.join(EMBED_DIR, f'alledge_{enc_type}_decoder.pt'))

    return test_aucs, n_param


# ============================================================
# 4. 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder', type=str, default='graphsage')
    args = parser.parse_args()

    encoders = [args.encoder] if args.encoder != 'all' else [
        'graphconv', 'graphsage', 'gat', 'film', 'gin', 'han', 'gatv2',
        'dualchannel', 'unified_dc']

    all_results = {}
    for enc in encoders:
        t0 = time.time()
        try:
            test_aucs, n_param = train_one_encoder(enc)
            elapsed = time.time() - t0
            all_results[enc] = {'aucs': test_aucs, 'params': n_param, 'time': elapsed}
            print(f"  [{enc}] 完成 ({elapsed:.0f}s, params={n_param:,})")
        except Exception as e:
            print(f"  [{enc}] 失败: {e}")
            import traceback; traceback.print_exc()

    # 保存汇总 JSON
    with open(RESULTS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {os.path.abspath(RESULTS_PATH)}")

    # 打印对比表
    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("  逐边 Test AUC 对比")
        print(f"{'='*70}")
        edge_keys = sorted(set().union(*(r['aucs'].keys() for r in all_results.values())) - {'_avg'})
        print(f"{'Encoder':<16s}", end='')
        for ek in edge_keys[:5]:
            print(f"  {ek[:25]:<25s}", end='')
        print(f"  {'AVG':>8s}")
        for enc in all_results:
            r = all_results[enc]['aucs']
            print(f"{enc:<16s}", end='')
            for ek in edge_keys[:5]:
                v = r.get(ek, '-')
                print(f"  {v:.4f}" if isinstance(v, float) else f"  {str(v):>25s}", end='')
            print(f"  {r['_avg']:8.4f}")


if __name__ == '__main__':
    main()
