# -*- coding: utf-8 -*-
"""Generate LaTeX table bodies for the revised manuscript from the new results."""
import os, json, statistics

DATA = os.path.join(os.path.dirname(__file__), '..', '数据')
R = json.load(open(os.path.join(DATA, 'multiseed_results.json'), encoding='utf-8'))
M = json.load(open(os.path.join(DATA, 'maxmatch_results.json'), encoding='utf-8'))
L, REC, SEEDS = R['link'], R['rec'], [str(s) for s in R['seeds']]
PARAMS, TIME = R.get('params', {}), R.get('time', {})

MAIN = ['graphconv', 'graphsage', 'gat', 'film', 'gin', 'han', 'gatv2', 'unified_dc']
NAME = {'graphconv': 'GraphConv', 'graphsage': 'GraphSAGE', 'gat': 'GAT',
        'film': 'FiLM', 'gin': 'GIN', 'han': 'HAN', 'gatv2': 'GATv2',
        'unified_dc': '\\DA{}', 'unified_dc_ch1only': 'Channel 1 only',
        'unified_dc_ch2only': 'Channel 2 only'}
EDGES = ['cert -> crew', 'college -> crew', 'region -> crew', 'crew -> vessel',
         'engine_type -> engine', 'vessel_type -> vessel',
         'auxiliary_equipment -> vessel', 'vessel -> owner']
EDGE_LABEL = {
    'cert -> crew': 'cert $\\to$ crew',
    'college -> crew': 'college $\\to$ crew',
    'region -> crew': 'region $\\to$ crew',
    'crew -> vessel': 'crew $\\to$ vessel',
    'engine_type -> engine': 'engine\\_type $\\to$ engine',
    'vessel_type -> vessel': 'vessel\\_type $\\to$ vessel',
    'auxiliary_equipment -> vessel': 'auxiliary\\_equipment $\\to$ vessel',
    'vessel -> owner': 'vessel $\\to$ owner',
}
METRICS7 = ['MaxMatch@1', 'MaxMatch@5', 'NDCG@10', 'MRR@10', 'HitRate@10', 'Precision@10']
OUT = {}


def ms(vals):
    a = [float(v) for v in vals]
    return statistics.mean(a), (statistics.stdev(a) if len(a) > 1 else 0.0)


def pm(vals, nd=3):
    m, s = ms(vals)
    return f'{m:.{nd}f}$\\pm${s:.{nd}f}'


def pm_dot(vals):
    m, s = ms(vals)
    a, b = f'{m:.3f}', f'{s:.3f}'
    return f'{a[1:]}$\\pm${b[1:]}'


def link(mk, e):
    return [L[mk][s][e] for s in SEEDS]


def rec(mk, m):
    return [REC[mk][s][m] for s in SEEDS]


def mm(mk, m):
    return [M[mk][s][m] for s in SEEDS]


def mean(vals):
    return statistics.mean(float(v) for v in vals)


def bold_best(cols, fmt=pm):
    means = {mk: mean(v) for mk, v in cols.items()}
    target = max(means.values())
    return {mk: (f'\\textbf{{{fmt(v)}}}' if abs(means[mk] - target) < 1e-12 else fmt(v))
            for mk, v in cols.items()}


# Table 5
order = sorted(MAIN, key=lambda mk: mean(link(mk, '_avg')), reverse=True)
rows = []
for rank, mk in enumerate(order, 1):
    a = pm(link(mk, '_avg'))
    p = PARAMS.get(mk, 0) / 1e6
    t = statistics.mean(TIME[mk][s] for s in SEEDS) if mk in TIME else 0
    if rank == 1:
        a = f'\\textbf{{{a}}}'
    rows.append(f'{rank} & {NAME[mk]} & {a} & {p:.2f}M & {t:.0f}s \\\\')
OUT['tbl:overall'] = '\n'.join(rows)

# Table 6
rows = []
for e in EDGES:
    cells = bold_best({mk: link(mk, e) for mk in MAIN})
    rows.append(EDGE_LABEL[e] + ' & ' + ' & '.join(cells[mk] for mk in MAIN) + ' \\\\')
cells = bold_best({mk: link(mk, '_avg') for mk in MAIN})
rows.append('\\midrule')
rows.append('\\textbf{Avg.} & ' + ' & '.join(cells[mk] for mk in MAIN) + ' \\\\')
OUT['tbl:peredfe'] = '\n'.join(rows)

# Table 7
cols = {m: {mk: (mm(mk, m) if m.startswith('MaxMatch') else rec(mk, m)) for mk in MAIN}
        for m in METRICS7}
cells = {m: bold_best(cols[m]) for m in METRICS7}
OUT['tbl:rec'] = '\n'.join(f'{NAME[mk]} & ' + ' & '.join(cells[m][mk] for m in METRICS7) + ' \\\\'
                           for mk in MAIN)

# Table 8
SUB = ['graphsage', 'film', 'unified_dc']
MS = ['MRR@5', 'MRR@10', 'MRR@50', 'NDCG@5', 'NDCG@10', 'NDCG@50']
cells = {m: bold_best({mk: rec(mk, m) for mk in SUB}) for m in MS}
OUT['tbl:mrrndcg'] = '\n'.join(f'{NAME[mk]} & ' + ' & '.join(cells[m][mk] for m in MS) + ' \\\\'
                               for mk in SUB)

# Table 9
KS = [5, 10, 20, 50, 100]
METS = [f'HitRate@{k}' for k in KS] + [f'Precision@{k}' for k in KS]
cells = {m: bold_best({mk: rec(mk, m) for mk in MAIN}, fmt=pm_dot) for m in METS}
OUT['tbl:hp'] = '\n'.join(f'{NAME[mk]} & ' + ' & '.join(cells[m][mk] for m in METS) + ' \\\\'
                          for mk in MAIN)

# Table 10
ABL = ['unified_dc_ch1only', 'unified_dc_ch2only', 'unified_dc']
ABL_NAME = {'unified_dc_ch1only': 'Channel 1 only',
            'unified_dc_ch2only': 'Channel 2 only', 'unified_dc': 'Full'}
ABLM = ['FieldPrec@10', 'StrictMatch@10', 'NDCG@10', 'MRR@10', 'HitRate@10', 'Precision@10']
auc_cells = bold_best({mk: link(mk, '_avg') for mk in ABL})
cells = {m: bold_best({mk: rec(mk, m) for mk in ABL}, fmt=lambda v: pm(v, 4)) for m in ABLM}
OUT['tbl:ablation'] = '\n'.join(
    f'{ABL_NAME[mk]} & {auc_cells[mk]} & {PARAMS.get(mk, 0)/1e6:.2f}M & '
    + ' & '.join(cells[m][mk] for m in ABLM) + ' \\\\' for mk in ABL)

json.dump(OUT, open(os.path.join(DATA, 'table_rows.json'), 'w', encoding='utf-8'),
          ensure_ascii=False, indent=1)
print('table_rows.json written')
