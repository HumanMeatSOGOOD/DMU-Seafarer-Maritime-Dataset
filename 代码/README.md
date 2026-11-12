# Code directory

| File | Purpose |
|---|---|
| `DualChannelGAT.py` | Dual-channel graph attention layers, dual identity projection, commonality abstraction, gating and fusion. |
| `hetero_graph_directed.py` | Builds the part-to-whole SMN `HeteroData` graph from the 19 edge tables. |
| `all_edge_train.py` | Trains one encoder on all 19 relations; link-prediction evaluation. |
| `multiseed_run.py` | Runs 10 model variants x 5 seeds and writes `multiseed_results.json`. |
| `multiseed_stats.py` | Prints five-seed mean/std tables from `multiseed_results.json`. |
| `recommendation_realistic.py` | Generates the rule-based recruitment queries and evaluates available embeddings. |
| `eval_maxmatch.py` | MaxMatch/Perfect evaluation. |
| `make_tables.py` | Generates LaTeX table bodies from the result JSON files. |
| `plot_all_methods_k_curves.py` | Generates the all-method K-curve figure and CSV. |
| `plot_da_k_curves.py` | Generates the Dualism-Attention-only K-curve figure. |

## Conventions

- All scripts resolve data relative to `../数据`; run them from the repository
  root or from `代码/` without changing the directory names.
- The code and comments use Chinese; this does not affect execution.
- `torch.load(..., weights_only=False)` is required for PyG `HeteroData`.
- Training outputs go to `数据/节点嵌入/`; figures go to `图片/`.

## Quick commands

```bash
python 代码/make_tables.py
python 代码/plot_all_methods_k_curves.py
python 代码/all_edge_train.py --encoder unified_dc
python 代码/multiseed_run.py
```

See `../docs/REPRODUCIBILITY.md` for details and runtime notes.
