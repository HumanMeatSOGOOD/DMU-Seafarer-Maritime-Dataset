# Headline results

All numbers below are five-seed means (seeds 42, 1, 2, 3, 4). Full-precision
records are in `数据/multiseed_results.json` and
`数据/maxmatch_results.json`.

## Link prediction

| Setting | Result |
|---|---|
| All-edge average AUC, Dualism-Attention | **0.716** |
| Best baseline (GraphSAGE) | 0.725 |
| Gap to best | 0.009 AUC |
| Parameter budget | **4.01M** (smallest among the eight methods) |
| Rank among eight methods | 5th, within 0.009 of the best |
| Core `crew -> vessel` edge | **0.840 AUC** |
| Strongest baseline on `crew -> vessel` | 0.700 AUC |
| Margin on `crew -> vessel` | **+0.140 AUC (20%)** |
| Crew attribute edges | within 0.6 AUC points of the best |

## Recommendation at K = 10

| Metric | Dualism-Attention |
|---|---:|
| MaxMatch@1 | **0.140** |
| MaxMatch@5 | 0.306 |
| NDCG@10 | 0.652 |
| MRR@10 | 0.029 |
| HitRate@10 | **0.104** |
| Precision@10 | 0.012 |

Dualism-Attention ranks first on all six metrics at K = 10. The K-sensitivity
analysis in the manuscript reports HitRate@100 = 0.512 for Dualism-Attention;
the complete curves are in `数据/all_methods_k_curves.csv`.

## Ablation (all-edge average AUC)

| Variant | AUC |
|---|---:|
| Channel 1 only | 0.703 |
| Channel 2 only | 0.577 |
| Full Dualism-Attention | **0.716** |

The ablation shows that both channels are necessary; Channel 1 is the main
driver of link prediction, while Channel 2 provides the commonality view.

## Where the numbers come from

| File | Contents |
|---|---|
| `数据/multiseed_results.json` | Per-method, per-seed link AUC and recommendation metrics. |
| `数据/maxmatch_results.json` | MaxMatch/Perfect records. |
| `数据/all_methods_k_curves.csv` | Metric/method/K mean and standard deviation. |
| `数据/da_k_curves.csv` | Dualism-Attention-only K curves. |
| `数据/table_rows.json` | LaTeX table bodies generated from the results. |
| `数据/final_tables.txt` | Rendered table text used in the manuscript. |

## Reproducing the tables and figures

Without retraining:

```bash
python 代码/make_tables.py
python 代码/plot_all_methods_k_curves.py
```

With retraining, see `docs/REPRODUCIBILITY.md`.
