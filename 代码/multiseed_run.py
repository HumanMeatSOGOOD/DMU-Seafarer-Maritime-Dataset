"""
多种子重跑: 10 模型 (8 方法 + 2 消融变体) × 5 种子。
每个种子保存嵌入 (alledge_{method}_seed{S}_embeddings.pt) + 链路 AUC;
推荐评测在固定查询集上对每个种子嵌入分别进行 (方差仅来自训练)。
保存 数据/multiseed_results.json, 打印 mean±std 摘要。
"""
import os, sys, json, time, io, contextlib
sys.path.insert(0, os.path.dirname(__file__))
import torch
torch.use_deterministic_algorithms(True)   # 消除 scatter_add 等非确定性, 保证可复现
import torch.nn.functional as F
import recommendation_realistic as rr
from all_edge_train import train_one_encoder, EMBED_DIR, DATA_DIR, GRAPH_PATH

SEEDS = [42, 1, 2, 3, 4]
METHODS = ['graphconv', 'graphsage', 'gat', 'film', 'gin', 'han', 'gatv2',
           'unified_dc', 'unified_dc_ch1only', 'unified_dc_ch2only']
KS = [5, 10, 20, 50, 100]
DISPLAY = {'graphconv': 'GraphConv', 'graphsage': 'GraphSAGE', 'gat': 'GAT',
           'film': 'FiLM', 'gin': 'GIN', 'han': 'HAN', 'gatv2': 'GATv2',
           'unified_dc': 'UnifiedDC', 'unified_dc_ch1only': 'Ch1only',
           'unified_dc_ch2only': 'Ch2only'}

THS = [('L', lambda L: L), ('Lm1', lambda L: max(1, L - 1)),
       ('Lm2', lambda L: max(1, L - 2)), ('Lm3', lambda L: max(1, L - 3))]


@contextlib.contextmanager
def suppress_stdout():
    old = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old


def eval_all(emb, queries, matrices, ks):
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
        fp = sm = mrr = ndcg = hr1 = pr1 = 0.0
        for i in range(nq):
            tk = topk[i, :k]; fields = queries[i]['fields']; L = Ls[i]
            matched = 0
            for atype, idx in fields:
                hits = matrices[atype][tk, idx] > 0
                if hits.sum() > 0:
                    matched += 1
            fp += matched / L; sm += 1.0 if matched == L else 0.0
            crew_rel = torch.zeros(k)
            for atype, idx in fields:
                crew_rel += matrices[atype][tk, idx]
            pos = torch.arange(1, k + 1, dtype=torch.float32)
            dcg = (crew_rel / torch.log2(pos + 1)).sum()
            ideal, _ = crew_rel.sort(descending=True)
            idcg = (ideal / torch.log2(pos + 1)).sum()
            ndcg += (dcg / (idcg + 1e-8)).item()
            rel = mcs[i] >= max(1, L - 1); R = rel.sum().item()
            if R > 0:
                f = rel[tk].sum().item()
                hr1 += 1.0 if f > 0 else 0.0
                pr1 += f / k
                # MRR: rank of the first relevant candidate (same L-1 relevance as HR/P)
                if f > 0:
                    mrr += 1.0 / (torch.where(rel[tk])[0][0].item() + 1)
        n = nq
        res[f'FieldPrec@{k}'] = fp / n; res[f'StrictMatch@{k}'] = sm / n
        res[f'MRR@{k}'] = mrr / n; res[f'NDCG@{k}'] = ndcg / n
        res[f'HitRate@{k}'] = hr1 / n; res[f'Precision@{k}'] = pr1 / n
    # 阈值敏感性 @10
    for tn, th in THS:
        hr = pr = 0.0
        for i in range(nq):
            rel = mcs[i] >= th(Ls[i]); R = rel.sum().item(); tk = topk[i, :10]
            if R > 0:
                f = rel[tk].sum().item()
                hr += 1.0 if f > 0 else 0.0
                pr += f / 10
        res[f'HR10_{tn}'] = hr / nq; res[f'P10_{tn}'] = pr / nq
    return res


def main():
    print('=' * 70)
    print(f'  多种子重跑: {len(METHODS)} 模型 × {len(SEEDS)} 种子 = {len(METHODS)*len(SEEDS)} 次训练')
    print('=' * 70)
    data = torch.load(GRAPH_PATH, weights_only=False)
    matrices, _ = rr.build_attribute_matrices(data)
    queries = torch.load(rr.QUERY_PATH, weights_only=False)

    link, rec = {}, {}
    params, times = {}, {}
    out_path = os.path.join(DATA_DIR, 'multiseed_results.json')

    # 断点续跑: 如果已有结果, 跳过已经完成的 (方法, 种子)
    if os.path.exists(out_path):
        old = json.load(open(out_path, encoding='utf-8'))
        for mk, d in old.get('link', {}).items():
            link[mk] = {int(k): v for k, v in d.items()}
        for mk, d in old.get('rec', {}).items():
            rec[mk] = {int(k): v for k, v in d.items()}
        params = old.get('params', {})
        times = {mk: {int(k): v for k, v in d.items()} for mk, d in old.get('time', {}).items()}
        print(f'  断点续跑: 已有 {sum(len(v) for v in link.values())} 个 (方法, 种子) 结果')

    def save():
        json.dump({'link': {mk: {str(k): v for k, v in d.items()} for mk, d in link.items()},
                   'rec': {mk: {str(k): v for k, v in d.items()} for mk, d in rec.items()},
                   'seeds': list(SEEDS), 'params': params,
                   'time': {mk: {str(k): v for k, v in d.items()} for mk, d in times.items()}},
                  open(out_path, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=2)

    t_start = time.time()
    for mk in METHODS:
        link.setdefault(mk, {}); rec.setdefault(mk, {}); times.setdefault(mk, {})
        for s in SEEDS:
            emb_path = os.path.join(EMBED_DIR, f'alledge_{mk}_seed{s}_embeddings.pt')
            if s in link[mk] and s in rec[mk] and os.path.exists(emb_path):
                print(f'  {DISPLAY[mk]:<11s} seed{s:<2d} | 已完成, 跳过')
                continue
            t0 = time.time()
            with suppress_stdout():
                aucs, n_param = train_one_encoder(mk, seed=s, save_suffix=f'seed{s}')
                emb = torch.load(os.path.join(EMBED_DIR, f'alledge_{mk}_seed{s}_embeddings.pt'),
                                 weights_only=False)
                rec[mk][s] = eval_all(emb, queries, matrices, KS)
            link[mk][s] = aucs
            params[mk] = n_param
            times[mk][s] = time.time() - t0
            print(f'  {DISPLAY[mk]:<11s} seed{s:<2d} | AUC={aucs["_avg"]:.4f} '
                  f'FP@10={rec[mk][s]["FieldPrec@10"]:.4f} HR@10={rec[mk][s]["HitRate@10"]:.4f} '
                  f'({time.time()-t0:.0f}s)')
            sys.stdout.flush()
            save()   # 每跑完一个 (方法, 种子) 就落盘, 长跑中断不丢结果

    save()
    print(f'\n总耗时 {time.time()-t_start:.0f}s. 已保存 multiseed_results.json')


if __name__ == '__main__':
    main()
