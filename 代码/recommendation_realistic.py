"""
=============================================================================
海员推荐实验 —— 真实招聘信息模拟版
=============================================================================

与 recommendation_experiment.py 的区别:
  查询不再从 7 类字段均匀独立采样, 而是模拟真实招聘逻辑生成:
    1. 以一艘真实船 V 为锚点 (船侧字段值全部取自 V, 保证组合现实可行)
    2. 随机选职务 cert → 判定部门 (甲板部 / 轮机部, 电气部并入轮机部)
    3. 部门决定核心船侧字段: 甲板部→船型/吨位; 轮机部→主机类型/型号/辅机/其他设备
    4. 个人字段 (籍贯/毕业院校) 按全局船员频率加权随机点缀
    5. 设备字段 (辅机/其他设备) 随机补充
    6. 字段数约束在 [4,7], 过滤无效查询

输出 (均独立存储, 不覆盖原实验):
  数据/推荐实验结果/realistic_queries.pt        机器格式查询 (供评测)
  数据/推荐实验结果/realistic_queries_sample.json 人类可读样本 (前 100 条)
  数据/推荐实验结果_真实招聘.md                  对比报告
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os, json, time
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 0. 配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '数据')
GRAPH_PATH = os.path.join(DATA_DIR, 'hetero_graph_directed.pt')
EDGE_DIR = os.path.join(DATA_DIR, '边数据')
EMBED_DIR = os.path.join(DATA_DIR, '节点嵌入')
OUTPUT_DIR = os.path.join(DATA_DIR, '推荐实验结果')
REPORT_PATH = os.path.join(DATA_DIR, '推荐实验结果_真实招聘.md')
QUERY_PATH = os.path.join(OUTPUT_DIR, 'realistic_queries.pt')
SAMPLE_PATH = os.path.join(OUTPUT_DIR, 'realistic_queries_sample.json')

# 单边训练嵌入 (8 种)
METHODS_ORDER = ['graphconv', 'graphsage', 'gat', 'film', 'gin', 'han', 'gatv2', 'dualchannel']
METHOD_NAMES = {
    'graphconv': 'GraphConv', 'graphsage': 'GraphSAGE', 'gat': 'GAT',
    'film': 'FiLM', 'gin': 'GIN', 'han': 'HAN', 'gatv2': 'GATv2',
    'dualchannel': 'DualChannelGAT',
}
# 全边训练嵌入 (9 种, 含 UnifiedDC)
ALLEDGE_METHODS = ['graphconv', 'graphsage', 'gat', 'film', 'gin', 'han', 'gatv2',
                   'dualchannel', 'unified_dc']
ALLEDGE_NAMES = {
    'graphconv': 'GraphConv', 'graphsage': 'GraphSAGE', 'gat': 'GAT',
    'film': 'FiLM', 'gin': 'GIN', 'han': 'HAN', 'gatv2': 'GATv2',
    'dualchannel': 'DualChannel', 'unified_dc': 'UnifiedDC',
}

NUM_QUERIES = 10000
K_VALUES = [5, 10, 20, 50, 100]
SEED = 42

# ---- 证书 → 部门 ----
# 图里的 cert 索引 = 原始编号按字符串排序后的位置 (pos1, pos10, pos11, ...),
# 不能假设成 pos1..pos14 的数字顺序; 名称直接读节点表, 部门按名称判定。
def _load_cert_names():
    import pandas as pd
    fp = os.path.join(DATA_DIR, '节点数据', '证书职务.xlsx')
    df = pd.read_excel(fp)
    name = dict(zip(df['类别编号'].astype(str), df['证书职务名称'].astype(str)))
    order = sorted(name)          # 与 hetero_graph_directed.py 的节点排序一致
    return {i: name[pos] for i, pos in enumerate(order)}

CERT_NAMES = _load_cert_names()
DECK_NAMES = {'船长', '大副', '二副', '三副', '高级值班水手', '值班水手'}
DECK_CERTS = {i for i, n in CERT_NAMES.items() if n in DECK_NAMES}

# ---- 字段优先级 (用于裁剪到 ≤7 时丢弃; 越大越保留) ----
PRIO = {
    'cert': 100, 'voyage': 90,
    'vessel_type': 80, 'tonnage_type': 80, 'engine_type': 80, 'engine': 80,
    'auxiliary_equipment': 80, 'other_equipment': 80,           # 作为核心时
    'region': 60, 'college': 60,
    'other_equip_type': 40,
}
# 设备补充字段 (低优先级, 裁剪时先丢)
SUPP_PRIO = {'auxiliary_equipment': 40, 'other_equipment': 40, 'other_equip_type': 40}


# ============================================================
# 1. 扩展属性矩阵 (11 类字段)
# ============================================================
def ei_to_sparse(edge_index, num_src, num_dst):
    idx = edge_index
    val = torch.ones(edge_index.shape[1])
    return torch.sparse_coo_tensor(idx, val, (num_src, num_dst)).coalesce()


def build_attribute_matrices(data):
    """构建 11 类字段的 crew×attr 二值矩阵 (dense float)。"""
    num_crew = data['crew'].num_nodes
    num_vessel = data['vessel'].num_nodes
    matrices = {}

    # ---- 1-hop 直接属性 ----
    for atype, et in [('cert', ('cert', 'attribute_of', 'crew')),
                      ('college', ('college', 'attribute_of', 'crew')),
                      ('region', ('region', 'attribute_of', 'crew'))]:
        ei = data[et].edge_index
        n_attr = data[et[0]].num_nodes
        m = torch.zeros(num_crew, n_attr)
        m[ei[1], ei[0]] = 1.0
        matrices[atype] = m

    # crew → vessel (part→whole): [N_crew, N_vessel]
    crew_vessel = ei_to_sparse(data[('crew', 'serves_on', 'vessel')].edge_index,
                               num_crew, num_vessel)

    def vessel_attr_matrix(et, n_attr):
        """crew → vessel ← attr : 翻转 attr→vessel 得 vessel→attr, 再乘 crew_vessel。"""
        ei = data[et].edge_index               # [attr, vessel]
        ve = torch.sparse_coo_tensor(
            ei.flip(0), torch.ones(ei.shape[1]), (num_vessel, n_attr)
        ).coalesce()
        return (torch.sparse.mm(crew_vessel, ve).to_dense() > 0).float()

    matrices['vessel_type'] = vessel_attr_matrix(
        ('vessel_type', 'attribute_of', 'vessel'), data['vessel_type'].num_nodes)
    matrices['tonnage_type'] = vessel_attr_matrix(
        ('tonnage_type', 'attribute_of', 'vessel'), data['tonnage_type'].num_nodes)
    matrices['engine'] = vessel_attr_matrix(
        ('engine', 'part_of', 'vessel'), data['engine'].num_nodes)
    matrices['auxiliary_equipment'] = vessel_attr_matrix(
        ('auxiliary_equipment', 'part_of', 'vessel'), data['auxiliary_equipment'].num_nodes)
    matrices['other_equipment'] = vessel_attr_matrix(
        ('other_equipment', 'part_of', 'vessel'), data['other_equipment'].num_nodes)
    matrices['voyage'] = vessel_attr_matrix(
        ('voyage', 'sailed_by', 'vessel'), data['voyage'].num_nodes)

    # ---- 3-hop: crew → vessel ← engine ← engine_type ----
    ei_eng = data[('engine', 'part_of', 'vessel')].edge_index
    ve = torch.sparse_coo_tensor(
        ei_eng.flip(0), torch.ones(ei_eng.shape[1]),
        (num_vessel, data['engine'].num_nodes)).coalesce()
    ei_et = data[('engine_type', 'attribute_of', 'engine')].edge_index
    e_et = torch.sparse_coo_tensor(
        ei_et.flip(0), torch.ones(ei_et.shape[1]),
        (data['engine'].num_nodes, data['engine_type'].num_nodes)).coalesce()
    matrices['engine_type'] = (torch.sparse.mm(
        torch.sparse.mm(crew_vessel, ve), e_et).to_dense() > 0).float()

    # ---- 3-hop: crew → vessel ← other_equipment ← other_equip_type ----
    ei_oe = data[('other_equipment', 'part_of', 'vessel')].edge_index
    v_oe = torch.sparse_coo_tensor(
        ei_oe.flip(0), torch.ones(ei_oe.shape[1]),
        (num_vessel, data['other_equipment'].num_nodes)).coalesce()
    ei_ot = data[('other_equip_type', 'attribute_of', 'other_equipment')].edge_index
    oe_ot = torch.sparse_coo_tensor(
        ei_ot.flip(0), torch.ones(ei_ot.shape[1]),
        (data['other_equipment'].num_nodes, data['other_equip_type'].num_nodes)).coalesce()
    matrices['other_equip_type'] = (torch.sparse.mm(
        torch.sparse.mm(crew_vessel, v_oe), oe_ot).to_dense() > 0).float()

    attr_info = {
        'cert': 14, 'college': 13, 'region': 32,
        'vessel_type': 12, 'tonnage_type': 3,
        'engine_type': 5, 'engine': 409,
        'auxiliary_equipment': 60, 'other_equipment': 458,
        'other_equip_type': 39, 'voyage': 1892,
    }
    return matrices, attr_info


# ============================================================
# 2. 每船属性抽取 + 完整船集合 + 个人字段频率
# ============================================================
def build_vessel_attrs(data):
    """为每艘船汇总其属性 (用于查询生成时取值)。"""
    va = defaultdict(lambda: {
        'vessel_type': None, 'tonnage_type': None,
        'engine': None, 'engine_type': None,
        'aux': None, 'voyages': [], 'other_equips': [], 'other_equip_types': [],
    })

    def collect(et, key, multi=False):
        ei = data[et].edge_index  # [attr, vessel]
        for a, v in zip(ei[0].tolist(), ei[1].tolist()):
            if multi:
                va[v][key].append(a)
            else:
                va[v][key] = a

    collect(('vessel_type', 'attribute_of', 'vessel'), 'vessel_type')
    collect(('tonnage_type', 'attribute_of', 'vessel'), 'tonnage_type')
    collect(('engine', 'part_of', 'vessel'), 'engine')
    collect(('auxiliary_equipment', 'part_of', 'vessel'), 'aux')
    collect(('voyage', 'sailed_by', 'vessel'), 'voyages', multi=True)
    collect(('other_equipment', 'part_of', 'vessel'), 'other_equips', multi=True)

    # engine → engine_type
    eng_to_type = {}
    ei = data[('engine_type', 'attribute_of', 'engine')].edge_index  # [etype, engine]
    for et, eng in zip(ei[0].tolist(), ei[1].tolist()):
        eng_to_type[eng] = et
    # other_equipment → other_equip_type
    oe_to_type = {}
    ei = data[('other_equip_type', 'attribute_of', 'other_equipment')].edge_index
    for ot, oe in zip(ei[0].tolist(), ei[1].tolist()):
        oe_to_type[oe] = ot

    for v, d in va.items():
        if d['engine'] is not None:
            d['engine_type'] = eng_to_type.get(d['engine'])
        d['other_equip_types'] = sorted({oe_to_type[oe] for oe in d['other_equips']
                                         if oe in oe_to_type})
    return va


def select_complete_vessels(data, va):
    """完整船: 同时有 vessel_type, tonnage, voyage, engine, aux。"""
    ei_cv = data[('crew', 'serves_on', 'vessel')].edge_index
    has_crew = set(ei_cv[1].tolist())
    out = []
    for v, d in va.items():
        if (d['vessel_type'] is not None and d['tonnage_type'] is not None
                and d['voyages'] and d['engine'] is not None
                and d['aux'] is not None and v in has_crew):
            out.append(v)
    return sorted(out)


def build_personal_freq(data):
    """籍贯 / 毕业院校 的全局船员持有频率 (用于加权采样)。"""
    def freq(et):
        ei = data[et].edge_index  # [attr, crew]
        c = Counter(ei[0].tolist())
        n = data[et[0]].num_nodes
        f = torch.zeros(n)
        for k, v in c.items():
            f[k] = v
        return f
    return freq(('region', 'attribute_of', 'crew')), freq(('college', 'attribute_of', 'crew'))


# ============================================================
# 3. 真实招聘查询生成
# ============================================================
def _rand(p):
    return torch.rand(1).item() < p


def _pick(lst):
    return lst[torch.randint(0, len(lst), (1,)).item()]


def generate_realistic_queries(matrices, va, complete_vessels,
                               region_freq, college_freq, num_queries, seed=SEED):
    torch.manual_seed(seed)
    num_crew = matrices['cert'].shape[0]
    queries = []
    attempts = 0
    max_attempts = num_queries * 30
    dept_counter = Counter()
    field_counter = Counter()

    while len(queries) < num_queries and attempts < max_attempts:
        attempts += 1
        V = complete_vessels[torch.randint(0, len(complete_vessels), (1,)).item()]
        d = va[V]
        c = torch.randint(0, 14, (1,)).item()
        is_deck = c in DECK_CERTS
        dept_counter['甲板部' if is_deck else '轮机部'] += 1

        # fields: list of (type, idx, prio)
        fields = [('cert', c, PRIO['cert'])]
        present = {'cert'}

        # 航线 (船 V 的实际航线)
        if _rand(0.75):
            fields.append(('voyage', _pick(d['voyages']), PRIO['voyage']))
            present.add('voyage')

        # 部门核心字段
        if is_deck:
            if _rand(0.85):
                fields.append(('vessel_type', d['vessel_type'], PRIO['vessel_type']))
                present.add('vessel_type')
            if _rand(0.55):
                fields.append(('tonnage_type', d['tonnage_type'], PRIO['tonnage_type']))
                present.add('tonnage_type')
        else:  # 轮机部 (含电气)
            et = _rand(0.85)
            eng = _rand(0.40)
            use_et = et and d['engine_type'] is not None   # 部分船主机缺类型, 守卫
            if use_et:
                fields.append(('engine_type', d['engine_type'], PRIO['engine_type']))
                present.add('engine_type')
            elif eng:
                fields.append(('engine', d['engine'], PRIO['engine']))
                present.add('engine')
            if _rand(0.70):
                fields.append(('auxiliary_equipment', d['aux'],
                               SUPP_PRIO['auxiliary_equipment']))
                present.add('auxiliary_equipment')
            if _rand(0.50) and d['other_equips']:
                fields.append(('other_equipment', _pick(d['other_equips']),
                               SUPP_PRIO['other_equipment']))
                present.add('other_equipment')

        # 设备补充 (未出现时)
        if 'auxiliary_equipment' not in present and _rand(0.30):
            fields.append(('auxiliary_equipment', d['aux'], SUPP_PRIO['auxiliary_equipment']))
            present.add('auxiliary_equipment')
        if 'other_equipment' not in present and d['other_equips'] and _rand(0.30):
            fields.append(('other_equipment', _pick(d['other_equips']),
                           SUPP_PRIO['other_equipment']))
            present.add('other_equipment')
        if ('other_equipment' not in present and 'other_equip_type' not in present
                and d['other_equip_types'] and _rand(0.20)):
            fields.append(('other_equip_type', _pick(d['other_equip_types']),
                           SUPP_PRIO['other_equip_type']))
            present.add('other_equip_type')

        # 个人字段 (全局频率加权)
        if _rand(0.40):
            r = torch.multinomial(region_freq, 1).item()
            fields.append(('region', r, PRIO['region']))
            present.add('region')
        if _rand(0.30):
            co = torch.multinomial(college_freq, 1).item()
            fields.append(('college', co, PRIO['college']))
            present.add('college')

        # 约束到 [4, 7]
        fields = _clamp_fields(fields, d, region_freq, college_freq)

        # 有效性: 至少 1 个 crew 满足全部字段
        mask = torch.ones(num_crew)
        ok = True
        for atype, idx, _ in fields:
            if idx is None:
                ok = False
                break
            mask = mask * matrices[atype][:, idx]
        if not ok:
            continue
        if mask.sum() > 0:
            clean = [(t, i) for t, i, _ in fields]
            for t, _ in clean:
                field_counter[t] += 1
            queries.append({'fields': clean, 'num_valid_crew': int(mask.sum().item())})

    stats = {
        'attempts': attempts, 'valid_rate': len(queries) / attempts,
        'dept': dict(dept_counter), 'field_freq': dict(field_counter),
    }
    return queries, stats


def _clamp_fields(fields, d, region_freq, college_freq):
    """裁剪/补齐到 [4, 7] 个字段。"""
    present = {f[0] for f in fields}

    # 补齐到 4 (按优先级顺序)
    fill = [
        ('region', lambda: torch.multinomial(region_freq, 1).item()),
        ('college', lambda: torch.multinomial(college_freq, 1).item()),
        ('other_equipment', lambda: _pick(d['other_equips']) if d['other_equips'] else None),
        ('auxiliary_equipment', lambda: d['aux']),
        ('tonnage_type', lambda: d['tonnage_type']),
        ('other_equip_type', lambda: _pick(d['other_equip_types']) if d['other_equip_types'] else None),
        ('engine', lambda: d['engine']),
        ('vessel_type', lambda: d['vessel_type']),
        ('voyage', lambda: _pick(d['voyages']) if d['voyages'] else None),
    ]
    i = 0
    while len(fields) < 4 and i < len(fill):
        t, fn = fill[i]; i += 1
        if t in present:
            continue
        v = fn()
        if v is None:
            continue
        fields.append((t, v, PRIO.get(t, 40)))
        present.add(t)

    # 裁剪到 7 (丢最低优先级; 同优先级丢最后加入的)
    while len(fields) > 7:
        # 按 (prio 升序, 加入顺序降序) 排序, 删第一个
        order = sorted(range(len(fields)),
                       key=lambda i: (fields[i][2], -i))
        del fields[order[0]]
    return fields


# ============================================================
# 4. 评测 (与原实验同构, 支持 11 类字段)
# ============================================================
def mean_pool_query(embeddings, fields):
    vecs = [embeddings[atype][node_idx] for atype, node_idx in fields]
    v = torch.stack(vecs).mean(dim=0)
    return F.normalize(v, dim=-1, eps=1e-8)


def evaluate_method(embeddings, crew_emb_norm, queries, matrices, ks):
    num_queries = len(queries)
    num_crew = crew_emb_norm.shape[0]

    gt_masks, gt_counts = [], []
    for q in queries:
        mask = torch.ones(num_crew)
        for atype, node_idx in q['fields']:
            mask = mask * matrices[atype][:, node_idx]
        gt_masks.append(mask.bool())
        gt_counts.append(int(mask.sum().item()))

    query_vecs = torch.stack([mean_pool_query(embeddings, q['fields']) for q in queries])
    scores = query_vecs @ crew_emb_norm.T
    topk_indices = torch.topk(scores, max(ks), dim=-1).indices

    results = {}
    for k in ks:
        topk = topk_indices[:, :k]
        fp_sum = sm_sum = mrr_sum = rec_sum = hr_sum = prec_sum = ndcg_sum = 0.0
        total_fields = 0
        for qi, q in enumerate(queries):
            tk = topk[qi]
            fields = q['fields']
            nf = len(fields)
            total_fields += nf
            matched = 0
            for atype, idx in fields:
                hits = matrices[atype][tk, idx] > 0
                if hits.sum() > 0:
                    matched += 1
                    mrr_sum += 1.0 / (torch.where(hits)[0][0].item() + 1)
            fp_sum += matched / nf
            sm_sum += 1.0 if matched == nf else 0.0

            crew_rel = torch.zeros(k)
            for atype, idx in fields:
                crew_rel += matrices[atype][tk, idx]
            pos = torch.arange(1, k + 1, dtype=torch.float32)
            dcg = (crew_rel / torch.log2(pos + 1)).sum()
            ideal, _ = crew_rel.sort(descending=True)
            idcg = (ideal / torch.log2(pos + 1)).sum()
            ndcg_sum += (dcg / (idcg + 1e-8)).item()

            gt = gt_masks[qi]
            gc = gt_counts[qi]
            if gc > 0:
                found = gt[tk].sum().item()
                rec_sum += found / gc
                hr_sum += 1.0 if found > 0 else 0.0
                prec_sum += found / k

        n = num_queries
        results[f'FieldPrec@{k}'] = fp_sum / n
        results[f'StrictMatch@{k}'] = sm_sum / n
        results[f'MRR@{k}'] = mrr_sum / total_fields
        results[f'Recall@{k}'] = rec_sum / n
        results[f'HitRate@{k}'] = hr_sum / n
        results[f'Precision@{k}'] = prec_sum / n
        results[f'NDCG@{k}'] = ndcg_sum / n
    return results


# ============================================================
# 5. 人类可读样本 (重建 index→orig_id)
# ============================================================
EDGE_FILES = [
    ("船员和船舶的关系.xlsx", "crew", "vessel"),
    ("船舶和船东的关系.xlsx", "vessel", "owner"),
    ("主机和主机生产厂商的关系.xlsx", "engine", "engine_factory"),
    ("辅机和辅机生产厂商的关系.xlsx", "auxiliary_equipment", "aux_equip_factory"),
    ("其他设备和其他设备生产厂商的关系.xlsx", "other_equipment", "other_equip_factory"),
    ("港口和国家的关系.xlsx", "port", "nation"),
    ("港口和航线的关系.xlsx", "port", "voyage"),
    ("船员和毕业院校的关系.xlsx", "crew", "college"),
    ("船员和籍贯地的关系.xlsx", "crew", "region"),
    ("船员和证书职务的关系.xlsx", "crew", "cert"),
    ("船舶和船舶类型的关系.xlsx", "vessel", "vessel_type"),
    ("船舶和吨位类型的关系.xlsx", "vessel", "tonnage_type"),
    ("船舶和主机的关系.xlsx", "vessel", "engine"),
    ("船舶和辅机的关系.xlsx", "vessel", "auxiliary_equipment"),
    ("船舶和其他设备的关系.xlsx", "vessel", "other_equipment"),
    ("船舶和航线的关系.xlsx", "vessel", "voyage"),
    ("主机和主机类型的关系.xlsx", "engine", "engine_type"),
    ("辅机和辅机类型的关系.xlsx", "auxiliary_equipment", "aux_equip_type"),
    ("其他设备和其他设备小类的关系.xlsx", "other_equipment", "other_equip_type"),
]


def build_id_to_idx_maps():
    """复刻 hetero_graph_directed.py 的节点映射, 返回 {type: {orig_id: idx}}。"""
    import pandas as pd
    all_ids = defaultdict(set)
    for fname, s, t in EDGE_FILES:
        fp = os.path.join(EDGE_DIR, fname)
        df = pd.read_excel(fp)
        all_ids[s].update(df['source'].astype(str).values)
        all_ids[t].update(df['target'].astype(str).values)
    maps = {}
    for nt, ids in all_ids.items():
        maps[nt] = {orig: i for i, orig in enumerate(sorted(ids))}
    # 反向: idx → orig_id
    inv = {nt: {i: orig for orig, i in m.items()} for nt, m in maps.items()}
    return inv


def make_readable_sample(queries, id_maps, n=100):
    sample = []
    for q in queries[:n]:
        fs = []
        for atype, idx in q['fields']:
            orig = id_maps.get(atype, {}).get(idx, str(idx))
            name = CERT_NAMES.get(idx, orig) if atype == 'cert' else orig
            fs.append({'type': atype, 'idx': idx, 'id': orig, 'name': name})
        sample.append({'fields': fs, 'num_valid_crew': q['num_valid_crew']})
    return sample


# ============================================================
# 6. 报告生成
# ============================================================
def generate_report(all_results, queries, stats, attr_info):
    lines = []
    lines.append("# 海员推荐实验结果 — 真实招聘信息模拟")
    lines.append("")
    lines.append(f"**实验日期**: {time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**查询数**: {len(queries)}")
    lines.append(f"**平均字段数**: {sum(len(q['fields']) for q in queries)/len(queries):.2f}")
    lines.append(f"**平均有效船员数**: {sum(q['num_valid_crew'] for q in queries)/len(queries):.1f}")
    lines.append(f"**评分方式**: 均值池化 + L2 归一化 + 余弦检索")
    lines.append("")

    lines.append("## 0. 查询生成统计 (真实招聘模拟)")
    lines.append("")
    lines.append(f"- 尝试次数: {stats['attempts']}, 有效率: {stats['valid_rate']*100:.1f}%")
    lines.append(f"- 部门分布: {stats['dept']}")
    lines.append("")
    lines.append("| 字段类型 | 出现次数 | 出现率 |")
    lines.append("|------|:---:|:---:|")
    nq = len(queries)
    for t in ['cert', 'voyage', 'vessel_type', 'tonnage_type', 'engine_type', 'engine',
              'auxiliary_equipment', 'other_equipment', 'other_equip_type', 'region', 'college']:
        c = stats['field_freq'].get(t, 0)
        lines.append(f"| {t} | {c} | {c/nq*100:.1f}% |")
    lines.append("")

    # 1. 核心对比
    lines.append("## 1. 核心对比: FieldPrec@10")
    lines.append("")
    lines.append("| 方法 | 单边训练 | 全边训练 | 变化 |")
    lines.append("|------|:---:|:---:|:---:|")
    best_s = best_a = None
    for mk in METHODS_ORDER:
        s = all_results.get(mk, {}).get('FieldPrec@10', 0)
        a = all_results.get(f'alledge_{mk}', {}).get('FieldPrec@10', 0)
        d = a - s
        best_s = max(best_s or (mk, s), (mk, s), key=lambda x: x[1])
        best_a = max(best_a or (mk, a), (mk, a), key=lambda x: x[1])
        ds = f"**+{d:+.3f}**" if d > 0 else f"{d:+.3f}"
        lines.append(f"| {METHOD_NAMES[mk]} | {s:.4f} | {a:.4f} | {ds} |")
    udc = all_results.get('alledge_unified_dc', {}).get('FieldPrec@10', 0)
    lines.append(f"| UnifiedDC | — | {udc:.4f} | — |")
    lines.append("")

    metric_names = ['FieldPrec', 'Recall', 'HitRate', 'Precision', 'StrictMatch', 'NDCG', 'MRR']

    # 2. 单边全指标
    lines.append("## 2. 全指标对比 @K=10 (单边训练)")
    lines.append("")
    lines.append("| 方法 | " + " | ".join(metric_names) + " |")
    lines.append("|------|" + "|".join(":---:" for _ in metric_names) + "|")
    for mk in METHODS_ORDER:
        r = all_results.get(mk, {})
        if not r: continue
        lines.append(f"| {METHOD_NAMES[mk]} | " + " | ".join(f"{r.get(f'{m}@10',0):.4f}" for m in metric_names) + " |")
    lines.append("")

    # 3. 全边全指标
    lines.append("## 3. 全指标对比 @K=10 (全边训练)")
    lines.append("")
    lines.append("| 方法 | " + " | ".join(metric_names) + " |")
    lines.append("|------|" + "|".join(":---:" for _ in metric_names) + "|")
    for mk in ALLEDGE_METHODS:
        r = all_results.get(f'alledge_{mk}', {})
        if not r: continue
        lines.append(f"| {ALLEDGE_NAMES[mk]} | " + " | ".join(f"{r.get(f'{m}@10',0):.4f}" for m in metric_names) + " |")
    lines.append("")

    # 4. K 值横扫
    lines.append("## 4. K 值横扫")
    lines.append("")
    all_show = ([(mk, '单边') for mk in METHODS_ORDER] +
                [(f'alledge_{mk}', '全边') for mk in METHODS_ORDER] +
                [('alledge_unified_dc', '全边')])
    for metric in metric_names:
        lines.append(f"### {metric}@K")
        lines.append("")
        lines.append("| 训练 | 方法 | " + " | ".join(f"K={k}" for k in K_VALUES) + " |")
        lines.append("|------|------|" + "|".join(":---:" for _ in K_VALUES) + "|")
        for mk, tt in all_show:
            r = all_results.get(mk, {})
            if not r: continue
            name = ALLEDGE_NAMES.get(mk.replace('alledge_', ''), METHOD_NAMES.get(mk, mk)) if mk.startswith('alledge_') else METHOD_NAMES.get(mk, mk)
            lines.append(f"| {tt} | {name} | " + " | ".join(f"{r.get(f'{metric}@{k}',0):.4f}" for k in K_VALUES) + " |")
        lines.append("")

    lines.append("## 5. 分析")
    lines.append("")
    lines.append("- 查询改为真实招聘模拟 (船锚点 + 部门条件采样 + 个人/设备字段点缀), 字段分布与组合更贴近现实。")
    lines.append(f"- 单边最优 FieldPrec@10: {METHOD_NAMES.get(best_s[0], best_s[0])} ({best_s[1]:.4f}); "
                 f"全边最优: {ALLEDGE_NAMES.get(best_a[0], best_a[0])} ({best_a[1]:.4f})。")
    lines.append("- 与均匀随机查询版 (见 推荐实验结果.md) 对比, 可观察方法排名是否稳健。")
    lines.append("")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


# ============================================================
# 7. 主流程
# ============================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("  海员推荐实验 — 真实招聘信息模拟")
    print("=" * 60)

    print("\n[1] 加载有向图 + 构建扩展属性矩阵 (11 类字段)...")
    data = torch.load(GRAPH_PATH, weights_only=False)
    matrices, attr_info = build_attribute_matrices(data)
    for atype, n in attr_info.items():
        cov = (matrices[atype].sum(dim=0) > 0).sum().item()
        print(f"  {atype:20s}: {n:4d} attr, {cov}/{n} 出现于 ≥1 crew")

    print("\n[2] 抽取每船属性 + 完整船集合 + 个人字段频率...")
    va = build_vessel_attrs(data)
    complete = select_complete_vessels(data, va)
    print(f"  完整船: {len(complete)}/{data['vessel'].num_nodes}")
    region_freq, college_freq = build_personal_freq(data)

    print(f"\n[3] 生成 {NUM_QUERIES} 条真实招聘查询...")
    queries, stats = generate_realistic_queries(
        matrices, va, complete, region_freq, college_freq, NUM_QUERIES, SEED)
    avg_f = sum(len(q['fields']) for q in queries) / len(queries)
    avg_v = sum(q['num_valid_crew'] for q in queries) / len(queries)
    print(f"  生成 {len(queries)} 条 (尝试 {stats['attempts']}, 有效率 {stats['valid_rate']*100:.1f}%)")
    print(f"  平均字段数: {avg_f:.2f}, 平均有效船员: {avg_v:.1f}")
    print(f"  部门分布: {stats['dept']}")
    print(f"  字段出现率: { {k: f'{v/len(queries)*100:.0f}%' for k,v in stats['field_freq'].items()} }")

    # 保存查询 (机器格式 + 可读样本)
    torch.save(queries, QUERY_PATH)
    print(f"\n  已保存查询 → {os.path.abspath(QUERY_PATH)}")
    id_maps = build_id_to_idx_maps()
    sample = make_readable_sample(queries, id_maps, n=100)
    with open(SAMPLE_PATH, 'w', encoding='utf-8') as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)
    print(f"  已保存可读样本 → {os.path.abspath(SAMPLE_PATH)}")
    print("\n  样例查询 (前 3 条):")
    for q in sample[:3]:
        print("    " + " + ".join(f"{f['type']}[{f['name']}]" for f in q['fields'])
              + f"  (valid_crew={q['num_valid_crew']})")

    # 评测
    all_results = {}
    print(f"\n[4] 评测单边训练嵌入 ({len(METHODS_ORDER)} 种)...")
    for mk in METHODS_ORDER:
        p = os.path.join(EMBED_DIR, f'{mk}_embeddings.pt')
        if not os.path.exists(p):
            print(f"  [SKIP] {METHOD_NAMES[mk]}: 缺失"); continue
        emb = torch.load(p, weights_only=False)
        ne = {nt: F.normalize(e, dim=-1, eps=1e-8) for nt, e in emb.items()}
        t0 = time.time()
        all_results[mk] = evaluate_method(ne, ne['crew'], queries, matrices, K_VALUES)
        print(f"  {METHOD_NAMES[mk]:16s} ({time.time()-t0:.1f}s): "
              f"FieldPrec@10={all_results[mk]['FieldPrec@10']:.4f}, "
              f"MRR@10={all_results[mk]['MRR@10']:.4f}")

    print(f"\n[5] 评测全边训练嵌入 ({len(ALLEDGE_METHODS)} 种)...")
    for mk in ALLEDGE_METHODS:
        p = os.path.join(EMBED_DIR, f'alledge_{mk}_embeddings.pt')
        if not os.path.exists(p):
            print(f"  [SKIP] {ALLEDGE_NAMES[mk]}: 缺失"); continue
        emb = torch.load(p, weights_only=False)
        ne = {nt: F.normalize(e, dim=-1, eps=1e-8) for nt, e in emb.items()}
        t0 = time.time()
        all_results[f'alledge_{mk}'] = evaluate_method(ne, ne['crew'], queries, matrices, K_VALUES)
        print(f"  {ALLEDGE_NAMES[mk]:16s} ({time.time()-t0:.1f}s): "
              f"FieldPrec@10={all_results[f'alledge_{mk}']['FieldPrec@10']:.4f}, "
              f"MRR@10={all_results[f'alledge_{mk}']['MRR@10']:.4f}")

    print(f"\n[6] 生成报告...")
    generate_report(all_results, queries, stats, attr_info)
    print(f"\nDone. 报告: {os.path.abspath(REPORT_PATH)}")


if __name__ == '__main__':
    main()
