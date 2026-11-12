"""
=============================================================================
有向异质图构建 — "整体-部分" (Whole-Part) 语义
=============================================================================

设计原则:
  所有边的方向统一为: 部分 (part) → 整体 (whole)
  例如:
    - cert (证书职务) → crew (船员)      证书是船员的一个属性
    - engine (主机) → vessel (船舶)      主机是船舶的一个零部件
    - vessel (船舶) → owner (船东)       船舶从属于船东
    - port (港口) → nation (国家)        港口属于国家

  图中不存在任何双向/反向边，天然是一个有向无环图 (DAG)。

与 hetero_graph_construct.py 的区别:
  - 后者: 19 种原始关系 + 19 种反向关系 = 38 种边 (双向)
  - 本文件: 仅 19 种有向边 (单向), 且方向统一为 part → whole

输出:
  数据/hetero_graph_directed.pt
=============================================================================
"""

import pandas as pd
import torch
from torch_geometric.data import HeteroData
from collections import defaultdict
import os

# ============================================================
# 0. 路径配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '数据')
EDGE_DIR = os.path.join(DATA_DIR, '边数据')

# ============================================================
# 1. 边配置: (Excel文件名, part类型, whole类型, 关系名, 是否需要翻转)
#    reverse=True 表示原始文件中 source→target 是 whole→part，
#    我们需要翻转成 part→whole。
# ============================================================

# 为了可读性, 分两组定义:

# --- A. 原始方向已是 part→whole, 无需翻转 ---
EDGES_KEEP = [
    # (文件名,                            part,        whole,          关系名)
    ("船员和船舶的关系.xlsx",              "crew",      "vessel",       "serves_on"),
    # crew (部分) → vessel (整体): 船员服务于船舶

    ("船舶和船东的关系.xlsx",              "vessel",    "owner",        "belongs_to"),
    # vessel (部分) → owner (整体): 船舶从属于船东

    ("主机和主机生产厂商的关系.xlsx",       "engine",    "engine_factory", "made_by"),
    # engine (部分) → factory (整体): 主机由厂商生产

    ("辅机和辅机生产厂商的关系.xlsx",       "auxiliary_equipment", "aux_equip_factory", "made_by"),
    # 辅机 → 工厂

    ("其他设备和其他设备生产厂商的关系.xlsx","other_equipment", "other_equip_factory", "made_by"),
    # 其他设备 → 工厂

    ("港口和国家的关系.xlsx",              "port",      "nation",       "located_in"),
    # port (部分) → nation (整体): 港口位于国家

    ("港口和航线的关系.xlsx",              "port",      "voyage",       "stop_of"),
    # port (部分) → voyage (整体): 港口是航线上的停靠点
]

# --- B. 原始方向是 whole→part, 需要翻转 ---
EDGES_REVERSE = [
    # (文件名,                            原始src(whole), 原始dst(part), 翻转后src(part), 翻转后dst(whole), 关系名)
    ("船员和毕业院校的关系.xlsx",           "crew",         "college",     "college",       "crew",           "attribute_of"),
    # 原始: crew→college. 翻转: college→crew. 院校是船员的一个属性

    ("船员和籍贯地的关系.xlsx",            "crew",         "region",      "region",        "crew",           "attribute_of"),
    # 原始: crew→region. 翻转: region→crew. 籍贯是船员的一个属性

    ("船员和证书职务的关系.xlsx",           "crew",         "cert",        "cert",          "crew",           "attribute_of"),
    # 原始: crew→cert. 翻转: cert→crew. 证书是船员的一个属性

    ("船舶和船舶类型的关系.xlsx",           "vessel",       "vessel_type", "vessel_type",   "vessel",         "attribute_of"),
    # 原始: vessel→vessel_type. 翻转: type→vessel. 类型是船舶的一个属性

    ("船舶和吨位类型的关系.xlsx",           "vessel",       "tonnage_type","tonnage_type",  "vessel",         "attribute_of"),
    # 原始: vessel→tonnage_type. 翻转: tonnage→vessel

    ("船舶和主机的关系.xlsx",              "vessel",       "engine",      "engine",        "vessel",         "part_of"),
    # 原始: vessel→engine. 翻转: engine→vessel. 主机是船舶的组成部分

    ("船舶和辅机的关系.xlsx",              "vessel",       "auxiliary_equipment",
                                                          "auxiliary_equipment", "vessel", "part_of"),
    # 原始: vessel→aux_equip. 翻转: aux_equip→vessel. 辅机是船舶的组成部分

    ("船舶和其他设备的关系.xlsx",           "vessel",       "other_equipment",
                                                          "other_equipment", "vessel",   "part_of"),
    # 原始: vessel→other_equip. 翻转: other_equip→vessel

    ("船舶和航线的关系.xlsx",              "vessel",       "voyage",      "voyage",        "vessel",         "sailed_by"),
    # 原始: vessel→voyage. 翻转: voyage→vessel. 航线包含船舶（船是航线的参与者）

    ("主机和主机类型的关系.xlsx",           "engine",       "engine_type", "engine_type",   "engine",         "attribute_of"),
    # 原始: engine→engine_type. 翻转: type→engine

    ("辅机和辅机类型的关系.xlsx",           "auxiliary_equipment", "aux_equip_type",
                                                          "aux_equip_type", "auxiliary_equipment", "attribute_of"),
    # 原始: aux_equip→aux_equip_type. 翻转: type→aux_equip

    ("其他设备和其他设备小类的关系.xlsx",    "other_equipment", "other_equip_type",
                                                          "other_equip_type", "other_equipment", "attribute_of"),
    # 原始: other_equip→other_equip_type. 翻转: type→other_equip
]


def build_node_mappings():
    """
    扫描所有边文件，建立节点 ID 映射。
    与双向图版本逻辑相同 —— 因为翻转边不改变节点集合。
    """
    all_ids = defaultdict(set)

    for config in EDGES_KEEP:
        filename, src_type, dst_type, _ = config
        filepath = os.path.join(EDGE_DIR, filename)
        df = pd.read_excel(filepath)
        all_ids[src_type].update(df['source'].astype(str).values)
        all_ids[dst_type].update(df['target'].astype(str).values)

    for config in EDGES_REVERSE:
        filename, orig_src, orig_dst, part_type, whole_type, _ = config
        filepath = os.path.join(EDGE_DIR, filename)
        df = pd.read_excel(filepath)
        # 翻转后 source 列是 part, target 列是 whole
        # 但原始列: source=orig_src(whole), target=orig_dst(part)
        # 数据行数相同, 集合不变, 只是方向反了
        all_ids[part_type].update(df['target'].astype(str).values)    # 原 target → 新 source (part)
        all_ids[whole_type].update(df['source'].astype(str).values)   # 原 source → 新 target (whole)

    node_mappings = {}
    node_counts = {}
    for ntype, ids in sorted(all_ids.items()):
        id_list = sorted(ids)
        node_mappings[ntype] = {orig: i for i, orig in enumerate(id_list)}
        node_counts[ntype] = len(id_list)

    return node_mappings, node_counts


def dedup_edges(edge_index):
    """去掉完全重复的 (u, v) 边：重复边会被随机分到 msg/test 两侧，造成泄漏。"""
    n_raw = edge_index.shape[1]
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index, n_raw


def build_hetero_data(node_mappings, node_counts):
    """构建有向无环的 whole-part 异质图"""
    data = HeteroData()

    # ---- 注册节点数量 ----
    for ntype, count in node_counts.items():
        data[ntype].num_nodes = count

    # ---- A 组: 原始方向已正确, 直接读取 ----
    print("\n  [A] 保持原方向的边 (part→whole):")
    for filename, part_type, whole_type, rel_name in EDGES_KEEP:
        filepath = os.path.join(EDGE_DIR, filename)
        df = pd.read_excel(filepath)

        part_map = node_mappings[part_type]
        whole_map = node_mappings[whole_type]

        part_indices = df['source'].astype(str).map(part_map).values
        whole_indices = df['target'].astype(str).map(whole_map).values

        edge_index = torch.tensor(
            [part_indices.tolist(), whole_indices.tolist()], dtype=torch.long
        )
        edge_index, n_raw = dedup_edges(edge_index)
        data[part_type, rel_name, whole_type].edge_index = edge_index
        print(f"    ({part_type:25s}) -[{rel_name:15s}]-> ({whole_type:25s}): "
              f"{n_raw:,} rows -> {edge_index.shape[1]:,} unique edges")

    # ---- B 组: 原始方向与 part→whole 相反, 交换 source 和 target ----
    print("\n  [B] 翻转方向的边 (whole→part → part→whole):")
    for filename, orig_src, orig_dst, part_type, whole_type, rel_name in EDGES_REVERSE:
        filepath = os.path.join(EDGE_DIR, filename)
        df = pd.read_excel(filepath)

        part_map = node_mappings[part_type]
        whole_map = node_mappings[whole_type]

        # 关键: 原始文件 source 是 whole, target 是 part
        # 翻转后: source 应是 part (原 target), target 应是 whole (原 source)
        part_indices = df['target'].astype(str).map(part_map).values    # 原 target → 新 source
        whole_indices = df['source'].astype(str).map(whole_map).values  # 原 source → 新 target

        edge_index = torch.tensor(
            [part_indices.tolist(), whole_indices.tolist()], dtype=torch.long
        )
        edge_index, n_raw = dedup_edges(edge_index)
        data[part_type, rel_name, whole_type].edge_index = edge_index
        print(f"    ({part_type:25s}) -[{rel_name:15s}]-> ({whole_type:25s}): "
              f"{n_raw:,} rows -> {edge_index.shape[1]:,} unique edges  "
              f"(原方向: {orig_src}→{orig_dst})")

    # ---- 不添加任何反向边 ----
    return data


def print_graph_summary(data):
    """打印图摘要"""
    print("\n" + "=" * 70)
    print("有向 Whole-Part 异质图 — 摘要")
    print("=" * 70)
    print(f"\n节点类型: {len(data.node_types)} 种")
    for ntype in sorted(data.node_types):
        print(f"  {ntype:30s}: {data[ntype].num_nodes:>8,} nodes")
    print(f"\n边类型: {len(data.edge_types)} 种 (均为有向, 无反向边)")
    total = 0
    for src, rel, dst in sorted(data.edge_types):
        n = data[src, rel, dst].edge_index.shape[1]
        total += n
        print(f"  ({src:20s}, {rel:20s}, {dst:20s}): {n:>8,} edges")
    print(f"\n去重后总边数: {total:,}")
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("构建有向 Whole-Part 异质知识图谱")
    print("=" * 70)

    # Step 1
    print("\n[1/3] 扫描边文件，建立节点 ID 映射...")
    node_mappings, node_counts = build_node_mappings()
    print(f"  发现 {len(node_mappings)} 种节点类型, 总节点 {sum(node_counts.values()):,} 个")

    # Step 2
    print("\n[2/3] 构建有向 HeteroData (part → whole)...")
    data = build_hetero_data(node_mappings, node_counts)

    # Step 3
    print_graph_summary(data)

    # 保存
    save_path = os.path.join(DATA_DIR, 'hetero_graph_directed.pt')
    torch.save(data, save_path)
    print(f"\n已保存至: {os.path.abspath(save_path)}")

    return data


if __name__ == '__main__':
    data = main()
