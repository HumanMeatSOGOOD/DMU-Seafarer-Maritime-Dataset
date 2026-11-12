# Reproducibility notes

## Tested environment

The results in this package were produced with:

- Python 3.13.5
- PyTorch 2.7.1 (CPU build in our environment)
- PyTorch Geometric 2.6.1
- torch-scatter 2.1.2
- NumPy 2.3.1, pandas 2.3.0, Matplotlib 3.10.3, openpyxl 3.1.5

A CUDA build of PyTorch can be used; install the matching `torch-scatter`
wheel for your PyTorch/CUDA/Python combination.

## Directory assumptions

All scripts use paths relative to the repository root:

```text
代码/   code
数据/   data
图片/   generated figures (created by the plotting script)
```

The code hard-codes `数据/` and `代码/` as directory names. Keep them when
re-uploading; otherwise change `DATA_DIR` in the scripts.

## What can be reproduced without retraining

The five-seed records are included:

- `数据/multiseed_results.json` — link prediction and recommendation results
  for all methods and seeds.
- `数据/maxmatch_results.json` — MaxMatch/Perfect results.
- `数据/all_methods_k_curves.csv` — K-curve data.
- `数据/table_rows.json`, `数据/final_tables.txt` — table bodies.

Regenerate the table bodies and K-curves:

```bash
python 代码/make_tables.py
python 代码/plot_all_methods_k_curves.py
```

`make_tables.py` writes `数据/table_rows.json`; the plotting script writes
`数据/all_methods_k_curves.csv` and PDF/PNG files under `图片/`.

## Rebuilding SMN

The released graph can be loaded directly. To rebuild it from the 19 edge
tables:

```bash
python 代码/hetero_graph_directed.py
```

Input: `数据/边数据/*.xlsx`
Output: `数据/hetero_graph_directed.pt`

The script builds node mappings from the union of all edge endpoints, applies
the part -> whole direction (flipping tables whose original direction is
whole -> part), and creates a `HeteroData` object with node counts and edge
indices only.

## Training one encoder

```bash
python 代码/all_edge_train.py --encoder unified_dc
```

The available encoder names are `graphconv`, `graphsage`, `gat`, `film`, `gin`,
`han`, `gatv2`, `dualchannel`, and `unified_dc`.

Outputs:

- `数据/节点嵌入/alledge_<encoder>_embeddings.pt`
- `数据/节点嵌入/alledge_<encoder>_decoder.pt`
- `数据/alledge_link_results.json`

## Full five-seed run

```bash
python 代码/multiseed_run.py
```

This is the expensive step: 10 model variants x 5 seeds = 50 training runs.
It writes embeddings under `数据/节点嵌入/` and updates
`数据/multiseed_results.json` after each `(method, seed)` pair (resumable).
Back up the included `multiseed_results.json` before running if you want to
keep the released copy.

The script enables `torch.use_deterministic_algorithms(True)`. On CUDA, set:

```bash
# PowerShell
$env:CUBLAS_WORKSPACE_CONFIG=":4096:8"
# Bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

Seeds: `42, 1, 2, 3, 4`.

Methods: GraphConv, GraphSAGE, GAT, FiLM, GIN, HAN, GATv2, UnifiedDC, and the
two ablation variants (Channel 1 only, Channel 2 only).

## Recommendation benchmark

The fixed 10,000-query set is already included in
`数据/推荐实验结果/realistic_queries.pt`, so the reported recommendation
numbers can be inspected without regenerating queries.

`python 代码/recommendation_realistic.py` regenerates the query set and
evaluates any embeddings present in `数据/节点嵌入/`; missing methods are
skipped with `[SKIP]`. It is optional and is not required for the manuscript
tables.

## Expected outputs

- Table bodies: `数据/table_rows.json`
- Per-seed records: `数据/multiseed_results.json`
- MaxMatch records: `数据/maxmatch_results.json`
- K-curve data: `数据/all_methods_k_curves.csv`, `数据/da_k_curves.csv`
- Figures: `图片/all_methods_k_curves.pdf`, `图片/kcurve_*.pdf`

## Runtime

A single encoder on CPU can take tens of seconds to a few minutes for the
small graph; the full 50-run sweep is substantially longer and is best run on
a GPU. The exact runtime is hardware-dependent and is recorded in
`multiseed_results.json` under `time`.

## Known practical notes

- The graph is a `HeteroData` object; load it with
  `torch.load(..., weights_only=False)`.
- The code uses Chinese comments and directory names; this is intentional and
  does not affect execution.
- The raw commercial/personal records are not part of this package; only the
  anonymized graph, edge tables, queries, and aggregate results are released.
- If `torch-scatter` fails to install, use the official PyG wheel index that
  matches your PyTorch/CUDA version.
