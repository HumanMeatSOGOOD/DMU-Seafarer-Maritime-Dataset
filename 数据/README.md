# Data directory

This directory contains the released, anonymized SMN data and the aggregate
results used in the manuscript.

| Path | Description |
|---|---|
| `hetero_graph_directed.pt` | PyTorch Geometric `HeteroData` graph; 20 node types, 19 edge types, 23,784 nodes, 104,571 edges. |
| `node_type_counts.csv` | Node counts per type. |
| `edge_type_counts.csv` | Edge counts per `(source, relation, target)`. |
| `dataset_summary.json` | Compact dataset summary. |
| `边数据/` | 19 anonymized `source`/`target` edge tables in the original part-to-whole direction (or with a documented flip in the build script). |
| `节点数据/证书职务.xlsx` | Certificate/rank identifier-to-name mapping (`pos1`, ...). |
| `节点数据/节点总表.xlsx` | Global anonymized node identifiers and type prefixes. |
| `推荐实验结果/realistic_queries.pt` | Fixed 10,000-query recommendation benchmark. |
| `推荐实验结果/realistic_queries_sample.json` | First 100 queries, human-readable. |
| `节点嵌入/` | Empty placeholder; training writes embeddings here. |
| `multiseed_results.json` | Full per-method/per-seed link and recommendation results. |
| `maxmatch_results.json` | MaxMatch/Perfect results. |
| `all_methods_k_curves.csv` | K-curve mean/std data for all methods. |
| `da_k_curves.csv` | Dualism-Attention-only K-curve data. |
| `table_rows.json` | LaTeX table bodies generated from the results. |
| `final_tables.txt` | Rendered table text. |
| `final_table_rows.txt` | Alternative table row export. |

## Key points

- The graph contains **no node features**. Models initialize learnable
  type-specific embeddings; this is the experimental setting used in the paper.
- Edges are directed **part -> whole**; the graph is a DAG.
- Identifiers are anonymized (`c1`, `v1`, `pos6`, ...). No seafarer names,
  vessel names, IMO numbers, timestamps, or raw commercial records are included.
- The folder names are in Chinese because the released code expects them; do
  not rename them without updating `DATA_DIR` in the scripts.

## Loading the graph

```python
import torch

g = torch.load("数据/hetero_graph_directed.pt", weights_only=False)
print(g)
print(g.node_types)
print(g.edge_types)
print(g.num_nodes, g.num_edges)
```

## License

See `../DATA_LICENSE.md` for the data license and privacy restrictions.
