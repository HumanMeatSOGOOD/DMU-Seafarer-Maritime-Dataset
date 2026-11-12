"""
MaxMatch@K = 每查询取 top-K 里匹配字段最多的船员, 看其匹配查询几成字段, 对查询取平均。
直接衡量"推荐列表里最好的船员满足查询多少比例"。Perfect@K = top-K 含全字段命中船员的查询占比(严格)。
用已保存的多种子确定性嵌入计算, 无需重训。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F
import numpy as np
import recommendation_realistic as rr
from all_edge_train import EMBED_DIR, DATA_DIR, GRAPH_PATH

SEEDS = [42, 1, 2, 3, 4]
METHODS = ['graphconv', 'graphsage', 'gat', 'film', 'gin', 'han', 'gatv2', 'unified_dc']
NAMES = {'graphconv': 'GraphConv', 'graphsage': 'GraphSAGE', 'gat': 'GAT',
         'film': 'FiLM', 'gin': 'GIN', 'han': 'HAN', 'gatv2': 'GATv2',
         'unified_dc': 'UnifiedDC'}
KS = [1, 5, 10, 20, 50]


def eval_maxmatch(emb, queries, matrices, ks):
    N = emb['crew'].shape[0]
    ne = {nt: F.normalize(e, dim=-1, eps=1e-8) for nt, e in emb.items()}
    crew = ne['crew']
    nq = len(queries)
    mcs, Ls = [], []
    for q in queries:
        mc = torch.zeros(N)
        for atype, idx in q['fields']:
            mc += matrices[atype][:, idx]
        mcs.append(mc); Ls.append(len(q['fields']))
    qv = torch.stack([F.normalize(torch.stack([ne[a][i] for a, i in q['fields']]).mean(0), dim=-1)
                      for q in queries])
    topk = torch.topk(qv @ crew.T, max(ks), dim=-1).indices
    res = {}
    for k in ks:
        tot = 0.0; perfect = 0
        for i in range(nq):
            best = mcs[i][topk[i, :k]].max().item()
            tot += best / Ls[i]
            if best >= Ls[i]:
                perfect += 1
        res[f'MaxMatch@{k}'] = tot / nq
        res[f'Perfect@{k}'] = perfect / nq
    return res


def main():
    data = torch.load(GRAPH_PATH, weights_only=False)
    matrices, _ = rr.build_attribute_matrices(data)
    queries = torch.load(rr.QUERY_PATH, weights_only=False)
    out = {}
    for mk in METHODS:
        out[mk] = {}
        for s in SEEDS:
            emb = torch.load(os.path.join(EMBED_DIR, f'alledge_{mk}_seed{s}_embeddings.pt'),
                             weights_only=False)
            out[mk][s] = eval_maxmatch(emb, queries, matrices, KS)
            print(f'  {NAMES[mk]} seed{s}: ' + ' '.join(f'MM@{k}={out[mk][s][f"MaxMatch@{k}"]:.3f}' for k in KS))

    print('\n' + '=' * 72)
    print('  MaxMatch@K (mean±std)  —— top-K 最佳船员平均匹配查询几成字段')
    print('=' * 72)
    print(f"{'方法':<13s} | " + ' | '.join(f'MM@{k}' for k in KS))
    print('-' * 72)
    for mk in METHODS:
        cells = [f'{np.mean([out[mk][s][f"MaxMatch@{k}"] for s in SEEDS]):.3f}±{np.std([out[mk][s][f"MaxMatch@{k}"] for s in SEEDS], ddof=1):.3f}'
                 for k in KS]
        print(f"{NAMES[mk]:<13s} | " + ' | '.join(f'{c:>12s}' for c in cells))

    print('\n' + '=' * 72)
    print('  Perfect@K (mean±std)  —— top-K 含全字段命中船员的查询占比(严格)')
    print('=' * 72)
    print(f"{'方法':<13s} | " + ' | '.join(f'Perf@{k}' for k in KS))
    print('-' * 72)
    for mk in METHODS:
        cells = [f'{np.mean([out[mk][s][f"Perfect@{k}"] for s in SEEDS]):.3f}±{np.std([out[mk][s][f"Perfect@{k}"] for s in SEEDS], ddof=1):.3f}'
                 for k in KS]
        print(f"{NAMES[mk]:<13s} | " + ' | '.join(f'{c:>12s}' for c in cells))

    json.dump(out, open(os.path.join(DATA_DIR, 'maxmatch_results.json'), 'w'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
