# -*- coding: utf-8 -*-
"""Figures: read results/csv/all_datasets.csv and write SVG figures to
figures/.

Methods are shown in separate supervised / unsupervised rows (the two families
are traditionally compared separately in the literature):
- supervised (6): mRMR / LDA / LASSO / LR / MI / NHMC
- unsupervised (6): MVPCA / MCD / DPC / HOGSP / CAE / ETP

Three figure families, each individually switchable on the command line:
- fig_q_delta_<family>    : rows = datasets, cols = [Q(S1) | ΔQ] heatmaps
- fig_random_ref_<family> : row = family, cols = datasets, Q(S1) curves against
                           the random min-max band
- fig_criterion_<family>  : row = family, cols = datasets, sign-corrected
                           relΔJ(%) curves

Usage:
  python3 plot_results.py                      # all three families
  python3 plot_results.py --no-criterion       # only q_delta / random_ref
  python3 plot_results.py --only-criterion     # only criterion
  python3 plot_results.py --svg                # SVG output
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / 'results' / 'csv' / 'all_datasets.csv'
FIG_DIR = ROOT / 'figures'

DATASET_ORDER = ['IP', 'PU', 'SL']

# supervised / unsupervised groups (same as METHODS in
# src/algorithms/__init__.py; filtered at runtime against the methods actually
# present in the CSV, so methods that were not run do not appear)
sys.path.insert(0, str(ROOT))
try:
    from src.algorithms import METHODS as REG_METHODS
    SUP_SET = {n for n, t, _ in REG_METHODS if t == 'sup'}
    UNSUP_SET = {n for n, t, _ in REG_METHODS if t == 'unsup'}
except Exception:
    SUP_SET = {'MI', 'mRMR', 'LDA', 'LASSO', 'LR', 'NHMC'}
    UNSUP_SET = {'ETP', 'MVPCA', 'MCD', 'DPC', 'HOGSP', 'CAE'}
GROUPS = [('supervised', SUP_SET), ('unsupervised', UNSUP_SET)]
GROUP_LABEL = {'supervised': 'supervised', 'unsupervised': 'unsupervised'}

# 15 colours + 15 markers. Method colours are assigned dynamically in global
# order from the method list pre-read from the CSV, so they are decoupled from
# method names (unique for up to 15 methods, then reused cyclically).
METHOD_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                  '#c44e52', '#4878cf', '#808000', '#6a51a3', '#ff9896']
METHOD_MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*', 'd',
                  '>', '<', 'p', 'H', '8']

# Criterion direction: +1 = larger objective is better (ΔJ = J(S1) - J(S2) > 0
# means the first selection is better); -1 = smaller is better (NHMC only: its
# criterion J_n is a mean inter-class correlation, so ΔJ <= 0 holds by
# construction and the sign is flipped to match the "positive = first
# selection better" convention).
CRITERION_DIRECTION = {
    'mRMR': +1, 'MVPCA': +1, 'MCD': +1, 'DPC': +1,
    'LDA': +1, 'LASSO': +1, 'LR': +1, 'HOGSP': +1,
    'MI': +1, 'ETP': +1, 'CAE': +1, 'NHMC': -1,
}

FMT = 'png'   # switched to 'svg' by --svg

# dataset label -> random baseline cache prefix
# (cache/random/<prefix>_k{k}_q1.npy)
RAND_CACHE_NAME = {'IP': 'indian_pines', 'PU': 'paviau', 'SL': 'salinas'}

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 18,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16,
})

# ============================================================================
# Per-family plot parameters. Change sizes / line widths / fonts / colours here
# only, not inside the plot_* functions. Keys are grouped per figure family and
# are independent of each other.
# ============================================================================
PLOT_CFG = {
    # Q(S1) and ΔQ heatmaps (rows = datasets, fixed 4 columns)
    'q_delta': {
        'fig_w': 20,          # figure width (inches), 4 columns fixed
        'row_h': 3.4,         # row height per dataset; rows = number of datasets
        'dpi': 150,
        'suptitle_fs': 13,
        'cmap_q': 'YlGnBu',   # single-hue ramp for Q(S1)
        'cmap_d': 'RdBu_r',   # diverging red-blue for Δ (centred on 0)
        'rect_top': 0.97,     # top margin for tight_layout (leaves room for suptitle)
        # font system (all sizes derived from base_fs)
        'annot_fs': 11,       # heatmap cell annotation size
        'base_fs': 12,        # base size for x/y tick labels
        'title_delta': 2,     # subplot title size = base_fs + 2
        'label_delta': 4,     # axis label size    = base_fs + 4
        'cbar_delta': 2,      # colorbar size      = base_fs + 2
        'annot_bold': False,   # bold cell annotations
        'tick_bold': True,    # bold tick labels
        'fontname': 'Times New Roman',
    },
    # Q(S1) vs random baseline (rows = family, cols = datasets, own y axis)
    'random_ref': {
        'col_w': 6.2,         # column width per dataset
        'row_h': 5.0,         # row height per family
        'dpi': 300,
        'suptitle_fs': 13,
        'lw': 1.8, 'ms': 5,  # method curve line width / marker size
        'band_color': '#c0c0c0', 'band_alpha': 0.40,   # random min-max band
        'med_color': '#444444', 'med_lw': 1.8, 'med_ms': 5,
        'p97_color': '#444444', 'p97_lw': 1.6, 'p97_ms': 6,
        'grid_alpha': 0.3,
        'legend_fs': 12, 'legend_ncol': 2,
        'rect_top': 0.95,
    },
    # Criterion differential relΔJ(%) (rows = family, cols = datasets, single
    # file; own y axis per subplot because scales differ across datasets)
    'criterion': {
        'col_w': 6.2,
        'row_h': 5.0,
        'dpi': 300,
        'suptitle_fs': 13,
        'lw': 1.8, 'ms': 5,
        'zero_color': '#444444', 'zero_lw': 1.0,
        'grid_alpha': 0.3,
        'legend_fs': 11, 'legend_ncol': 2,
        'ylim_pad_frac': 0.08, 'ylim_pad_min': 0.5,
        'rect_top': 0.95,
    },
}


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def _read_csv(path):
    """Read a CSV, auto-detecting the encoding (utf-8-sig / utf-8 / gbk) so
    both the gbk files written on Windows and utf-8 files on Linux work."""
    import io
    raw = Path(path).read_bytes()
    text = None
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        raise ValueError(f'cannot detect encoding: {path}')
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise ValueError(f'CSV is empty or missing a header: {path}')
    return rows


def load_rows(path):
    """Read the main CSV. Returns rows / methods / ks / dsets / idx."""
    rows = _read_csv(path)
    methods = list(dict.fromkeys(r['Method'] for r in rows))
    ks = sorted({int(r['k']) for r in rows})
    dsets = [d for d in DATASET_ORDER if any(r['Dataset'] == d for r in rows)]
    idx = {(r['Dataset'], r['Method'], int(r['k'])): r for r in rows}
    return rows, methods, ks, dsets, idx


def load_crit_rows(path):
    """Read a criterion CSV (columns: Dataset, Method, k, J1, J2, ΔJ,
    relΔJ(%)). Returns rows / methods / ks / dsets / idx."""
    rows = _read_csv(path)
    methods = list(dict.fromkeys(r['Method'] for r in rows))
    ks = sorted({int(r['k']) for r in rows})
    dsets = [d for d in DATASET_ORDER if any(r['Dataset'] == d for r in rows)]
    idx = {(r['Dataset'], r['Method'], int(r['k'])): r for r in rows}
    return rows, methods, ks, dsets, idx


def group_methods(methods):
    """Split the methods present in the CSV into supervised / unsupervised
    using the registry."""
    sup = [m for m in methods if m in SUP_SET]
    unsup = [m for m in methods if m in UNSUP_SET]
    other = [m for m in methods if m not in SUP_SET and m not in UNSUP_SET]
    return [('supervised', sup), ('unsupervised', unsup)], other


# Global method order (same as METHODS in src/algorithms/__init__.py): the 6
# supervised methods first, then the 6 unsupervised ones. Colours and markers
# are assigned uniquely over this order across all methods, so the supervised
# and unsupervised families never share a legend style (mRMR and MVPCA no
# longer share colour and marker).
GLOBAL_METHOD_ORDER = ['MI', 'mRMR', 'LDA', 'LASSO', 'LR', 'NHMC',
                       'ETP', 'MVPCA', 'MCD', 'DPC', 'HOGSP', 'CAE']


def method_style_map(methods_seen):
    """Assign a unique (colour, marker) to every method present, following the
    global order."""
    order = [m for m in GLOBAL_METHOD_ORDER if m in methods_seen]
    order += [m for m in methods_seen if m not in GLOBAL_METHOD_ORDER]
    return {m: (METHOD_PALETTE[i % len(METHOD_PALETTE)],
                METHOD_MARKERS[i % len(METHOD_MARKERS)])
            for i, m in enumerate(order)}


def grid_of(rows, methods, ks, dsets, idx, field, fmt=float):
    """Return {dataset: methods x ks matrix}."""
    out = {}
    for d in dsets:
        M = np.full((len(methods), len(ks)), np.nan)
        for i, m in enumerate(methods):
            for j, k in enumerate(ks):
                r = idx.get((d, m, k))
                if r is not None and r[field] not in ('', 'nan'):
                    M[i, j] = fmt(r[field])
        out[d] = M
    return out


def _n_rows(groups):
    return sum(1 for _, ms in groups if len(ms) > 0)


def _style_heatmap_panel(ax, hm, title, cbar_label, cfg):
    """Apply title / axis label / tick / colorbar styling to one heatmap panel.

    All sizes derive from cfg['base_fs'] plus the *_delta keys, and font /
    bold options live in one place, so the Q and Δ panels do not duplicate the
    styling code.
    """
    bfs = cfg['base_fs']
    fn = cfg.get('fontname')
    fs_title = bfs + cfg['title_delta']
    fs_label = bfs + cfg['label_delta']
    fs_cbar = bfs + cfg['cbar_delta']
    title_kw = {'fontsize': fs_title, 'fontweight': 'bold'}
    label_kw = {'fontsize': fs_label, 'fontweight': 'bold'}
    if fn:
        title_kw['fontname'] = fn
        label_kw['fontname'] = fn
    ax.set_title(title, **title_kw)
    ax.set_xlabel('k', **label_kw)
    ax.set_ylabel('method', **label_kw)
    ax.tick_params(axis='both', labelsize=bfs)
    if cfg.get('tick_bold', True):
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight('bold')
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fs_cbar)
    cbar_kw = {'fontsize': fs_cbar, 'fontweight': 'bold'}
    if fn:
        cbar_kw['fontname'] = fn
    cbar.set_label(cbar_label, **cbar_kw)


def plot_q_delta(rows, groups, ks, dsets, idx):
    """One figure per method family: supervised / unsupervised.

    Each figure has rows = datasets and two columns = [Q(S1) | ΔQ]. Matrices
    come from grid_of, styling from _style_heatmap_panel.
    """
    qd_fmt = 'svg'
    cfg = PLOT_CFG['q_delta']
    annot_kws = {'size': cfg['annot_fs']}
    if cfg.get('annot_bold', True):
        annot_kws['fontweight'] = 'bold'

    # panel spec: (column offset, data field, colour map, colorbar label,
    #              title suffix, number format, diverging?)
    panel_specs = [
        (0, 'Q(S1)', cfg['cmap_q'], 'Q(S1)', 'Q(S1)', '.3f', False),
        (1, 'ΔQ', cfg['cmap_d'], 'Δ = Q(S1) − Q(S2)', 'ΔQ', '+.3f', True),
    ]

    for gname, gmethods0 in groups:
        if len(gmethods0) == 0:
            continue
        methods = [m for m in GLOBAL_METHOD_ORDER if m in gmethods0]
        methods += [m for m in gmethods0 if m not in GLOBAL_METHOD_ORDER]

        # two columns per family, same panel width as the combined version
        fig, axes = plt.subplots(len(dsets), 2,
                                 figsize=(cfg['fig_w'] / 2,
                                          cfg['row_h'] * len(dsets)))
        if len(dsets) == 1:
            axes = axes[None, :]

        for a, d in enumerate(dsets):
            grids = {field: grid_of(rows, methods, ks, [d], idx, field)[d]
                     for _, field, *_ in panel_specs}
            vmax = max(0.01, float(np.nanmax(np.abs(grids['ΔQ']))))
            for off, field, cmap, cbar_lab, suffix, fmt, diverg in panel_specs:
                ax = axes[a, off]
                heat_kw = dict(annot=True, fmt=fmt, annot_kws=annot_kws,
                               cmap=cmap, xticklabels=ks, yticklabels=methods,
                               cbar_kws={'label': cbar_lab})
                if diverg:                   # Δ panel: symmetric colours around 0
                    heat_kw.update(center=0, vmin=-vmax, vmax=vmax)
                hm = sns.heatmap(grids[field], ax=ax, **heat_kw)
                _style_heatmap_panel(
                    ax, hm, f'{d} · {GROUP_LABEL[gname]} — {suffix}',
                    cbar_lab, cfg)

        #fig.suptitle('Band selection diagnosis by method family: absolute '
        #             'accuracy Q(S1) and re-selection differential Δ',
        #             fontsize=cfg['suptitle_fs'])
        fig.tight_layout(rect=(0, 0, 1, cfg['rect_top']))
        out = FIG_DIR / f'fig_q_delta_{gname}.{qd_fmt}'
        fig.savefig(out, dpi=cfg['dpi'])
        plt.close(fig)
        print('saved', out)


def _random_band(d, ks, idx, methods):
    """Read the random accuracy distribution from cache/random and return
    min / max / median / p97.5 per k.

    On a cache miss it falls back to Q1_rand_med / Q1_rand_p975 from the CSV
    (min and max become NaN, so the corresponding fill_between breaks).
    """
    rmin, rmax, rmed, rp97 = [], [], [], []
    for k in ks:
        p = ROOT / 'cache' / 'random' / \
            f'{RAND_CACHE_NAME.get(d, d.lower())}_k{k}_q1.npy'
        if p.exists():
            a = np.load(p)
            rmin.append(float(a.min())); rmax.append(float(a.max()))
            rmed.append(float(np.median(a)))
            rp97.append(float(np.percentile(a, 97.5)))
        else:
            row = next((idx[(d, m, k)] for m in methods if (d, m, k) in idx),
                       None)
            rmed.append(_to_float(row['Q1_rand_med']) if row else float('nan'))
            rp97.append(_to_float(row['Q1_rand_p975']) if row else float('nan'))
            rmin.append(float('nan')); rmax.append(float('nan'))
    return (np.array(rmin), np.array(rmax), np.array(rmed), np.array(rp97))


def plot_random_ref(rows, groups, ks, dsets, idx):
    """One figure per method family (single row = the family, cols = datasets).

    Each subplot scales its own y axis.
    """
    rr_fmt = 'svg'
    cfg = PLOT_CFG['random_ref']
    styles = method_style_map([r['Method'] for r in rows])

    for gname, gmethods0 in groups:
        if len(gmethods0) == 0:
            continue
        methods = [m for m in GLOBAL_METHOD_ORDER if m in gmethods0]
        methods += [m for m in gmethods0 if m not in GLOBAL_METHOD_ORDER]
        Q = grid_of(rows, methods, ks, dsets, idx, 'Q(S1)')

        fig, axes = plt.subplots(1, len(dsets),
                                 figsize=(cfg['col_w'] * len(dsets),
                                          cfg['row_h']))
        axes = np.atleast_2d(axes)   # single row: normalise to shape (1, n_dsets)

        for a, d in enumerate(dsets):
            ax = axes[0, a]
            rmin, rmax, rmed, rp97 = _random_band(d, ks, idx, methods)
            ax.fill_between(ks, rmin, rmax, color=cfg['band_color'],
                            alpha=cfg['band_alpha'], lw=0,
                            label='random min-max')
            ax.plot(ks, rmed, 's--', color=cfg['med_color'],
                    lw=cfg['med_lw'], ms=cfg['med_ms'],
                    label='random median')
            ax.plot(ks, rp97, 'x:', color=cfg['p97_color'],
                    lw=cfg['p97_lw'], ms=cfg['p97_ms'],
                    label='random p97.5')
            for ii, m in enumerate(methods):
                color, marker = styles[m]
                ax.plot(ks, Q[d][ii], '-', color=color, marker=marker,
                        lw=cfg['lw'], ms=cfg['ms'], label=m)
            ax.axhline(0, color='gray', lw=0.8)
            ax.set_xlabel('k')
            ax.set_ylabel('Q(S1)')
            ax.set_title(f'{d} · {GROUP_LABEL[gname]}')
            ax.grid(alpha=cfg['grid_alpha'])
            ax.legend(fontsize=cfg['legend_fs'], ncol=cfg['legend_ncol'],
                      loc='best')
            vals = [v for v in np.concatenate(
                [Q[d].ravel(), rmin, rmax, rmed, rp97]) if np.isfinite(v)]
            if vals:
                lo, hi = float(min(vals)), float(max(vals))
                pad = max((hi - lo) * 0.06, 1e-3)
                ax.set_ylim(lo - pad, hi + pad)

        #fig.suptitle('Selected-band accuracy vs random selection by method '
        #             'family (reference only)', fontsize=cfg['suptitle_fs'])
        fig.tight_layout(rect=(0, 0, 1, cfg['rect_top']))
        out = FIG_DIR / f'fig_random_ref_{gname}.{rr_fmt}'
        fig.savefig(out, dpi=cfg['dpi'])
        plt.close(fig)
        print('saved', out)


def load_crit_rows_multi(paths):
    """Merge several criterion CSVs (the usual layout is one
    {ds}_criterion_values.csv per dataset). Returns rows / ks / dsets / idx."""
    all_rows, all_idx = [], {}
    for path in paths:
        rows, _, _, _, idx = load_crit_rows(path)
        all_rows.extend(rows)
        all_idx.update(idx)
    ks = sorted({int(r['k']) for r in all_rows})
    dsets = [d for d in DATASET_ORDER if any(r['Dataset'] == d
                                             for r in all_rows)]
    return all_rows, ks, dsets, all_idx


def plot_criterion(crit_rows, groups, ks, dsets, crit_idx):
    """One figure per method family (single row = the family, cols = datasets).

    The curve is the sign-corrected relΔJ(%) = ΔJ/J1 x 100, where the sign of
    ΔJ is flipped according to CRITERION_DIRECTION so that everything is seen
    from the "larger objective is better" side: positive means the first
    selection has the better criterion value, negative means the re-selection
    is better. Colours and markers are shared with fig_random_ref through
    method_style_map, so methods look the same in both figures. Each subplot
    has its own y axis; a subplot is left empty when a family has no data for
    that dataset.
    """
    cri_fmt = 'svg'
    cfg = PLOT_CFG['criterion']
    styles = method_style_map([r['Method'] for r in crit_rows])
    present = {r['Method'] for r in crit_rows}

    for gname, methods0 in groups:
        gmethods = [m for m in methods0 if m in present]
        if len(gmethods) == 0:
            continue
        methods = [m for m in GLOBAL_METHOD_ORDER if m in gmethods]
        methods += [m for m in gmethods if m not in GLOBAL_METHOD_ORDER]

        fig, axes = plt.subplots(1, len(dsets),
                                 figsize=(cfg['col_w'] * len(dsets),
                                          cfg['row_h']))
        axes = np.atleast_2d(axes)   # single row: normalise to shape (1, n_dsets)

        for a, d in enumerate(dsets):
            ax = axes[0, a]
            all_ys = []
            for m in methods:
                color, marker = styles[m]
                sign = CRITERION_DIRECTION.get(m, 1)
                ys = []
                for k in ks:
                    r = crit_idx.get((d, m, k))
                    if r is None or r['relΔJ(%)'] in ('', 'nan'):
                        ys.append(float('nan'))
                    else:
                        ys.append(sign * _to_float(r['relΔJ(%)']))
                all_ys.append(ys)
                ax.plot(ks, ys, '-', color=color, marker=marker,
                        lw=cfg['lw'], ms=cfg['ms'], label=m)
            ax.axhline(0, color=cfg['zero_color'], lw=cfg['zero_lw'],
                       ls='--')
            ax.set_xlabel('k')
            ax.set_ylabel(r'signed $\Delta J/J_1$ (%)')
            # keep subplot titles short (dataset and family only); the meaning
            # is carried by the suptitle and ylabel, which avoids horizontal
            # overlap in multi-column layouts.
            ax.set_title(f'{d} · {GROUP_LABEL[gname]}')
            ax.grid(alpha=cfg['grid_alpha'])
            ax.legend(fontsize=cfg['legend_fs'], ncol=cfg['legend_ncol'],
                      loc='best')
            vals = [v for ysi in all_ys for v in ysi if np.isfinite(v)]
            if vals:
                lo, hi = float(min(vals)), float(max(vals))
                pad = max((hi - lo) * cfg['ylim_pad_frac'],
                          cfg['ylim_pad_min'])
                ax.set_ylim(min(lo - pad, 0.0), max(hi + pad, 0.0))

        #fig.suptitle('Criterion differential by method family: signed '
        #             r'relative $\Delta J/J_1$ (positive = first selection '
        #             'better)', fontsize=cfg['suptitle_fs'])
        fig.tight_layout(rect=(0, 0, 1, cfg['rect_top']))
        out = FIG_DIR / f'fig_criterion_{gname}.{cri_fmt}'
        fig.savefig(out, dpi=cfg['dpi'])
        plt.close(fig)
        print('saved', out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default=str(CSV_PATH))
    ap.add_argument('--svg', action='store_true',
                    help='save SVG (default PNG)')
    ap.add_argument('--criterion-csv', default=None,
                    help='defaults to results/csv/*criterion_values.csv')
    # per-family switches: all on by default; --no-xxx disables one, --only-xxx
    # enables just one.
    for flag, name, helptext in [
            ('--q-delta', 'q_delta', 'Q(S1)/ΔQ heatmaps (fig_q_delta)'),
            ('--random-ref', 'random_ref',
             'random baseline curves (fig_random_ref)'),
            ('--criterion', 'criterion',
             'criterion differential curves (fig_criterion)')]:
        ap.add_argument(flag, action=argparse.BooleanOptionalAction,
                        default=True,
                        help=f'plot {helptext} (default: on)')
        ap.add_argument(f'--only-{name.replace("_", "-")}',
                        action='store_true',
                        help=f'plot only {helptext}, disable the other two')
    args = ap.parse_args()

    # --only-xxx: enable just that family, disable the rest
    only = [n for n in ('q_delta', 'random_ref', 'criterion')
            if getattr(args, f'only_{n}')]
    if only:
        enabled = {n: (n in only)
                   for n in ('q_delta', 'random_ref', 'criterion')}
    else:
        enabled = {'q_delta': args.q_delta,
                   'random_ref': args.random_ref,
                   'criterion': args.criterion}
    print('figure switches:', {k: ('on' if v else 'off')
                               for k, v in enabled.items()})
    if not any(enabled.values()):
        print('all figure families are disabled, nothing to do.')
        return

    global FMT
    if args.svg:
        FMT = 'svg'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    need_main = enabled['q_delta'] or enabled['random_ref']
    groups = None
    if need_main:
        p = Path(args.csv)
        if not p.exists():
            print(f'CSV not found: {p} (run run_all.py first)')
            return
        rows, methods, ks, dsets, idx = load_rows(p)
        groups, other = group_methods(methods)
        print(f'rows={len(rows)} ks={ks} dsets={dsets}')
        print('supervised  :', [m for _, ms in groups if _ == 'supervised'
                                for m in ms])
        print('unsupervised:', [m for _, ms in groups if _ == 'unsupervised'
                                for m in ms])
        if other:
            print('ungrouped (ignored):', other)
        if enabled['q_delta']:
            plot_q_delta(rows, groups, ks, dsets, idx)
        if enabled['random_ref']:
            plot_random_ref(rows, groups, ks, dsets, idx)

    if enabled['criterion']:
        crit_path = args.criterion_csv
        if crit_path and Path(crit_path).exists():
            crit_rows, _, crit_ks, crit_dsets, crit_idx = \
                load_crit_rows(crit_path)
            print(f'criterion csv: {crit_path} rows={len(crit_rows)} '
                  f'ks={crit_ks} dsets={crit_dsets}')
            if groups is None:
                groups = [('supervised', sorted(SUP_SET)),
                          ('unsupervised', sorted(UNSUP_SET))]
            plot_criterion(crit_rows, groups, crit_ks, crit_dsets, crit_idx)
        else:
            cands = sorted((ROOT / 'results' / 'csv')
                           .glob('*criterion_values.csv'))
            if cands:
                crit_rows, crit_ks, crit_dsets, crit_idx = \
                    load_crit_rows_multi(cands)
                print(f'merged {len(cands)} criterion CSVs: rows={len(crit_rows)} '
                      f'ks={crit_ks} dsets={crit_dsets}')
                if groups is None:
                    groups, _ = group_methods(
                        list(dict.fromkeys(r['Method']
                                           for r in crit_rows)))
                plot_criterion(crit_rows, groups, crit_ks, crit_dsets,
                               crit_idx)
            else:
                print('no criterion CSV found, skipping the criterion figure '
                      '(use --criterion-csv to point at one)')


if __name__ == '__main__':
    main()
