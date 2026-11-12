"""
从 multiseed_results.json 计算 mean±std, 按论文各表格式打印 (markdown 可粘贴)。
8 主方法用于表 3/4/7/9/10/11/12; 2 消融变体 + Full 用于表 13。
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '数据')
R = json.load(open(os.path.join(DATA_DIR, 'multiseed_results.json'), encoding='utf-8'))
link, rec, seeds = R['link'], R['rec'], R['seeds']

MAIN = ['graphconv', 'graphsage', 'gat', 'film', 'gin', 'han', 'gatv2', 'unified_dc']
NAMES = {'graphconv': 'GraphConv', 'graphsage': 'GraphSAGE', 'gat': 'GAT',
         'film': 'FiLM', 'gin': 'GIN', 'han': 'HAN', 'gatv2': 'GATv2',
         'unified_dc': 'UnifiedDC', 'unified_dc_ch1only': 'Ch1 only',
         'unified_dc_ch2only': 'Ch2 only'}
ABL = ['unified_dc_ch1only', 'unified_dc_ch2only', 'unified_dc']


def ms(vals):
    """mean±std (sample std, ddof=1)."""
    a = np.array(vals, dtype=float)
    return a.mean(), a.std(ddof=1)


def fmt(vals):
    m, s = ms(vals)
    return f'{m:.3f}±{s:.3f}'


def fmt4(vals):
    m, s = ms(vals)
    return f'{m:.4f}±{s:.4f}'


print('=' * 70)
print('表 3  全边链路预测总体结果 (avg AUC mean±std)')
print('=' * 70)
print(f"{'方法':<13s} | {'AUC':<15s}")
for mk in MAIN:
    aucs = [link[mk][str(s)]['_avg'] for s in seeds]
    print(f"{NAMES[mk]:<13s} | {fmt(aucs)}")

print('\n' + '=' * 70)
print('表 4  代表性边逐方法 AUC (mean±std)')
print('=' * 70)
EDGES = ['cert -> crew', 'college -> crew', 'region -> crew', 'crew -> vessel',
         'engine_type -> engine', 'vessel_type -> vessel',
         'auxiliary_equipment -> vessel', 'vessel -> owner']
print('edge'.ljust(26), ' '.join(f'{NAMES[m]:>12s}' for m in MAIN))
for e in EDGES:
    cells = [fmt([link[mk][str(s)][e] for s in seeds]) for mk in MAIN]
    print(e.replace(' -> ', '→').ljust(26), ' '.join(f'{c:>12s}' for c in cells))
cells = [fmt([link[mk][str(s)]['_avg'] for s in seeds]) for mk in MAIN]
print('Avg'.ljust(26), ' '.join(f'{c:>12s}' for c in cells))

print('\n' + '=' * 70)
print('表 7  推荐 @K=10 (mean±std)')
print('=' * 70)
M7 = ['FieldPrec@10', 'StrictMatch@10', 'NDCG@10', 'MRR@10', 'HitRate@10', 'Precision@10']
print(f"{'方法':<13s} | " + ' | '.join(f'{m.split("@")[0]:>10s}' for m in M7))
for mk in MAIN:
    cells = [fmt4([rec[mk][str(s)][m] for s in seeds]) for m in M7]
    print(f"{NAMES[mk]:<13s} | " + ' | '.join(f'{c:>10s}' for c in cells))

print('\n' + '=' * 70)
print('表 9  FieldPrec@K (mean±std)')
print('=' * 70)
print(f"{'方法':<13s} | " + ' | '.join(f'K={k}' for k in [5, 10, 20, 50, 100]))
for mk in MAIN:
    cells = [fmt4([rec[mk][str(s)][f'FieldPrec@{k}'] for s in seeds]) for k in [5, 10, 20, 50, 100]]
    print(f"{NAMES[mk]:<13s} | " + ' | '.join(f'{c:>10s}' for c in cells))

print('\n' + '=' * 70)
print('表 10  MRR@K & NDCG@K (GraphSAGE, FiLM, UnifiedDC)')
print('=' * 70)
SUB = ['graphsage', 'film', 'unified_dc']
print(f"{'方法':<13s} | " + ' | '.join(f'{m}' for m in ['MRR@5', 'MRR@10', 'MRR@50', 'NDCG@5', 'NDCG@10', 'NDCG@50']))
for mk in SUB:
    cells = [fmt4([rec[mk][str(s)][m] for s in seeds]) for m in ['MRR@5', 'MRR@10', 'MRR@50', 'NDCG@5', 'NDCG@10', 'NDCG@50']]
    print(f"{NAMES[mk]:<13s} | " + ' | '.join(f'{c:>10s}' for c in cells))

print('\n' + '=' * 70)
print('表 11  HR@K & P@K (L-1, mean±std)')
print('=' * 70)
print(f"{'方法':<13s} | " + ' | '.join(f'HR@{k}' for k in [5, 10, 20, 50, 100]) + ' || ' + ' | '.join(f'P@{k}' for k in [5, 10, 20, 50, 100]))
for mk in MAIN:
    hr = [fmt4([rec[mk][str(s)][f'HitRate@{k}'] for s in seeds]) for k in [5, 10, 20, 50, 100]]
    p = [fmt4([rec[mk][str(s)][f'Precision@{k}'] for s in seeds]) for k in [5, 10, 20, 50, 100]]
    print(f"{NAMES[mk]:<13s} | " + ' | '.join(f'{c:>10s}' for c in hr) + ' || ' + ' | '.join(f'{c:>10s}' for c in p))

print('\n' + '=' * 70)
print('表 12  阈值敏感性 @10 (mean±std)')
print('=' * 70)
print(f"{'方法':<13s} | " + ' | '.join(f'HR({t})' for t in ['L', 'Lm1', 'Lm2', 'Lm3']) + ' || ' + ' | '.join(f'P({t})' for t in ['L', 'Lm1', 'Lm2', 'Lm3']))
for mk in MAIN:
    hr = [fmt4([rec[mk][str(s)][f'HR10_{t}'] for s in seeds]) for t in ['L', 'Lm1', 'Lm2', 'Lm3']]
    p = [fmt4([rec[mk][str(s)][f'P10_{t}'] for s in seeds]) for t in ['L', 'Lm1', 'Lm2', 'Lm3']]
    print(f"{NAMES[mk]:<13s} | " + ' | '.join(f'{c:>10s}' for c in hr) + ' || ' + ' | '.join(f'{c:>10s}' for c in p))

print('\n' + '=' * 70)
print('表 13  消融 (mean±std)')
print('=' * 70)
print(f"{'变体':<11s} | {'AUC':<11s} | " + ' | '.join(f'{m.split("@")[0]}' for m in ['FieldPrec@10', 'StrictMatch@10', 'NDCG@10', 'MRR@10', 'HitRate@10', 'Precision@10']))
for mk in ABL:
    auc = fmt([link[mk][str(s)]['_avg'] for s in seeds])
    cells = [fmt4([rec[mk][str(s)][m] for s in seeds]) for m in ['FieldPrec@10', 'StrictMatch@10', 'NDCG@10', 'MRR@10', 'HitRate@10', 'Precision@10']]
    print(f"{NAMES[mk]:<11s} | {auc:<11s} | " + ' | '.join(f'{c:>10s}' for c in cells))

if 'params' in R:
    print('\n' + '=' * 70)
    print('参数量与平均训练时间')
    print('=' * 70)
    for mk in MAIN:
        p = R['params'].get(mk, 0) / 1e6
        t = np.mean([R['time'][mk][str(s)] for s in seeds])
        print(f'{NAMES[mk]:<13s} | {p:5.2f}M | {t:5.0f}s')
