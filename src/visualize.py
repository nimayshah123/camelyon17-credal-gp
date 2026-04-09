"""
visualize.py — Publication-quality figures for the Camelyon17 Credal GP paper.

Three figures are produced, matching the style of the UCI Wine and Breast
Cancer Wisconsin case studies from the same paper series.

Figure 1 — Domain Profiles + Expansion Function
    Panel A: Hospital clusters in PCA-2 space with OOD gap highlighted.
    Panel B: Horizontal bar chart of top-15 expansion features.
    Panel C: Heatmap of per-hospital mean values for top-8 expansion features.

Figure 2 — Epistemic Uncertainty: Single GP vs Credal GP
    Panel A: Single GP uncertainty band along top expansion feature.
    Panel B: Credal GP uncertainty band (visibly wider in OOD region).
    Panel C: Overlay comparison of single vs credal widths.
    Panel D: Scatter of credal width vs OOD distance with Pearson r.

Figure 3 — Domain Selection + Metrics Comparison
    Panel A: Bar chart ranking all 5 hospitals by expected credal width reduction.
    Panel B: Before/after credal width after adding best hospital.
    Panel C: Metrics comparison table (ECE, NLL, AUROC, FPR@95).
    Panel D: Greedy selection improvement curve.

Style is consistent with credal_sim.py and credal_large.py:
  - Dark axes, minimal spines, stat annotation boxes.
  - Colour palette defined in config.py.
  - Saved as both PNG (150 dpi) and PDF.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
def _apply_style():
    """Apply consistent rcParams matching the paper's visual style."""
    plt.rcParams.update({
        'font.family':         'DejaVu Sans',
        'font.size':           10,
        'axes.titlesize':      11,
        'axes.labelsize':      10,
        'axes.grid':           True,
        'grid.color':          '#888888',
        'grid.linestyle':      '--',
        'grid.alpha':          0.3,
        'xtick.labelsize':     9,
        'ytick.labelsize':     9,
        'legend.fontsize':     9,
        'legend.framealpha':   0.85,
        'figure.facecolor':    'white',
        'axes.facecolor':      '#FAFAFA',
        'savefig.dpi':         config.DPI,
        'savefig.bbox':        'tight',
    })


def _stat_box(ax, text, x=0.97, y=0.97, fontsize=8.5):
    """Add a framed annotation box in the corner of an axis."""
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor='white',
            edgecolor='#AAAAAA',
            alpha=0.9,
        ),
    )


def _save_figure(fig, name: str):
    """Save figure as PNG and PDF to config.OUTPUTS_DIR."""
    os.makedirs(config.OUTPUTS_DIR, exist_ok=True)
    for ext in ('png', 'pdf'):
        path = os.path.join(config.OUTPUTS_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=config.DPI, bbox_inches='tight')
    print(f"  Saved -> {os.path.join(config.OUTPUTS_DIR, name)}.{{png,pdf}}")


# ---------------------------------------------------------------------------
# Figure 1
# ---------------------------------------------------------------------------
def plot_figure1(
    features_dict: Dict[int, np.ndarray],
    expansions: Dict[int, float],
    top_feature_indices: List[int],
    n_top_bar: int = 15,
    n_top_heatmap: int = 8,
) -> plt.Figure:
    """
    Figure 1: Domain profiles and expansion function.

    Args
    ----
    features_dict     : {hospital_id: np.ndarray (N, D)} — PCA features.
    expansions        : {feature_idx: expansion_value} — all expansions.
    top_feature_indices: list of feature indices sorted by expansion desc.
    n_top_bar         : how many features to show in bar chart.
    n_top_heatmap     : how many features to show in heatmap.

    Returns
    -------
    fig : matplotlib Figure
    """
    _apply_style()
    fig = plt.figure(figsize=(16, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                            left=0.06, right=0.97, bottom=0.12, top=0.90)

    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])
    ax_C = fig.add_subplot(gs[2])

    pal  = config.PALETTE['hospitals']

    # ------------------------------------------------------------------
    # Panel A: Hospital clusters in PCA-2 space
    # ------------------------------------------------------------------
    ax_A.set_title('A — Hospital Domains in PCA Space', fontweight='bold', pad=6)

    rng = np.random.RandomState(config.RANDOM_SEED)
    max_pts_per_h = 300

    handles_A = []
    for h in range(config.N_HOSPITALS):
        feats  = features_dict[h]
        n_show = min(len(feats), max_pts_per_h)
        idx    = rng.choice(len(feats), size=n_show, replace=False)
        x_vals = feats[idx, 0]
        y_vals = feats[idx, 1]
        color  = pal[h]
        alpha  = 0.35 if h == config.HELD_OUT_HOSPITAL else 0.55
        ms     = 10   if h == config.HELD_OUT_HOSPITAL else 8
        marker = 'D'  if h == config.HELD_OUT_HOSPITAL else 'o'

        sc = ax_A.scatter(
            x_vals, y_vals,
            c=color, alpha=alpha, s=ms, marker=marker,
            edgecolors='none', zorder=3,
        )
        label = config.HOSPITAL_NAMES[h]
        handles_A.append(mpatches.Patch(color=color, label=label))

    # Shade OOD coverage gap region
    ood_feats = features_dict[config.HELD_OUT_HOSPITAL][:, 0]
    src_feats = np.concatenate([features_dict[h][:, 0]
                                 for h in config.SOURCE_HOSPITALS
                                 if h in features_dict])
    ood_center = np.mean(ood_feats)
    src_max    = np.max(src_feats)
    if ood_center > src_max:
        gap_lo = src_max
        gap_hi = ood_center + (ood_center - src_max) * 0.5
        ax_A.axvspan(gap_lo, gap_hi, alpha=0.12, color=config.PALETTE['ood'],
                     label='OOD coverage gap', zorder=1)

    ax_A.set_xlabel('PCA Component 1')
    ax_A.set_ylabel('PCA Component 2')
    ax_A.legend(handles=handles_A, loc='lower left',
                fontsize=7.5, markerscale=1.2)
    _stat_box(ax_A, f"N hospitals = {config.N_HOSPITALS}\nOOD = H{config.HELD_OUT_HOSPITAL}")

    # ------------------------------------------------------------------
    # Panel B: Horizontal bar chart of top-n expansion features
    # ------------------------------------------------------------------
    ax_B.set_title('B — Top Expansion Features', fontweight='bold', pad=6)

    # Sort and take top-n
    sorted_exp = sorted(
        [(k, v) for k, v in expansions.items() if not np.isnan(v)],
        key=lambda x: x[1], reverse=True
    )[:n_top_bar]

    feat_labels  = [f"PCA-{k}" for k, _ in sorted_exp]
    exp_vals     = [v for _, v in sorted_exp]
    y_pos        = np.arange(len(feat_labels))[::-1]   # flip: largest at top

    bar_colors = [
        config.PALETTE['credal'] if i < 3 else '#AAAACC'
        for i in range(len(feat_labels))
    ]

    bars = ax_B.barh(y_pos, exp_vals, color=bar_colors, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    ax_B.set_yticks(y_pos)
    ax_B.set_yticklabels(feat_labels, fontsize=8)
    ax_B.set_xlabel('Expansion  e_φ(z) = σ²_between / σ²_within')
    ax_B.axvline(1.0, color='#888888', linestyle='--', linewidth=0.8,
                 alpha=0.7, label='e=1 (neutral)')

    # Annotate top bar
    top_val = exp_vals[0]
    ax_B.text(top_val + 0.01 * max(exp_vals), y_pos[0],
              f' {top_val:.2f}', va='center', fontsize=8,
              color=config.PALETTE['credal'], fontweight='bold')

    _stat_box(ax_B, f"Top feature: PCA-{sorted_exp[0][0]}\ne = {sorted_exp[0][1]:.3f}")

    # ------------------------------------------------------------------
    # Panel C: Heatmap of per-hospital means for top features
    # ------------------------------------------------------------------
    ax_C.set_title('C — Per-Hospital Feature Means (top features)',
                   fontweight='bold', pad=6)

    top_n_feats = [k for k, _ in sorted_exp[:n_top_heatmap]]
    n_h         = config.N_HOSPITALS
    means_mat   = np.zeros((n_h, n_top_heatmap))

    for row, h in enumerate(range(n_h)):
        for col, f in enumerate(top_n_feats):
            means_mat[row, col] = np.mean(features_dict[h][:, f])

    # Z-score across hospitals for each feature
    means_norm = (means_mat - means_mat.mean(axis=0, keepdims=True)) \
                 / (means_mat.std(axis=0, keepdims=True) + 1e-8)

    cmap = LinearSegmentedColormap.from_list(
        'credal_heatmap',
        ['#2E4057', '#FAFAFA', '#E84855'],
        N=256,
    )
    im = ax_C.imshow(means_norm, aspect='auto', cmap=cmap,
                      interpolation='nearest', vmin=-2, vmax=2)

    ax_C.set_xticks(range(n_top_heatmap))
    ax_C.set_xticklabels([f"PC{f}" for f in top_n_feats],
                          rotation=45, ha='right', fontsize=8)
    ax_C.set_yticks(range(n_h))
    ax_C.set_yticklabels(
        [config.HOSPITAL_NAMES[h] for h in range(n_h)],
        fontsize=8
    )

    # Highlight OOD row
    ood_row = config.HELD_OUT_HOSPITAL
    for spine in ['bottom', 'top', 'left', 'right']:
        ax_C.spines[spine].set_visible(False)
    rect = plt.Rectangle(
        (-0.5, ood_row - 0.5), n_top_heatmap, 1,
        linewidth=2.0, edgecolor=config.PALETTE['ood'],
        facecolor='none', zorder=5
    )
    ax_C.add_patch(rect)

    plt.colorbar(im, ax=ax_C, shrink=0.7, pad=0.02,
                  label='Z-scored mean')
    _stat_box(ax_C, f"OOD hospital\nhighlighted", fontsize=7.5)

    fig.suptitle(
        'Camelyon17-WILDS: Domain Profiles and Expansion Function',
        fontsize=13, fontweight='bold', y=1.01
    )

    _save_figure(fig, 'figure1_domain_profiles')
    return fig


# ---------------------------------------------------------------------------
# Figure 2
# ---------------------------------------------------------------------------
def plot_figure2(
    features_dict: Dict[int, np.ndarray],
    labels_dict: Dict[int, np.ndarray],
    credal_gp,
    top_feature_idx: int,
    ood_distances: np.ndarray,
) -> plt.Figure:
    """
    Figure 2: Single GP vs Credal GP uncertainty comparison.

    Args
    ----
    features_dict   : {hospital_id: np.ndarray (N, D)}.
    labels_dict     : {hospital_id: np.ndarray (N,)}.
    credal_gp       : fitted CredalGP instance.
    top_feature_idx : int — which PCA dim is the GP x-axis.
    ood_distances   : np.ndarray (N_ood,) — OOD distances for scatter.

    Returns
    -------
    fig : matplotlib Figure
    """
    _apply_style()
    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.40,
                            left=0.06, right=0.97, bottom=0.13, top=0.88)

    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])
    ax_C = fig.add_subplot(gs[2])
    ax_D = fig.add_subplot(gs[3])

    # Evaluation grid
    all_x = np.concatenate([
        features_dict[h][:, top_feature_idx]
        for h in range(config.N_HOSPITALS)
        if h in features_dict
    ])
    x_min, x_max = all_x.min(), all_x.max()
    margin        = 0.15 * (x_max - x_min)
    X_grid        = np.linspace(x_min - margin, x_max + margin, 300)

    # Source region boundary (max of source hospital means)
    src_means = [
        np.mean(features_dict[h][:, top_feature_idx])
        for h in config.SOURCE_HOSPITALS
        if h in features_dict
    ]
    src_boundary = max(src_means)

    # GP predictions
    center_mean, lower_mean, upper_mean, credal_width = credal_gp.predict(X_grid)
    single_var = credal_gp.single_gp_variance(X_grid)
    single_mu  = credal_gp.single_gp_mean(X_grid)

    single_std = np.sqrt(np.maximum(single_var, 0))
    # Credal width for single GP is 0 by definition; use 2*std for band
    single_band_lo = single_mu - 2.0 * single_std
    single_band_hi = single_mu + 2.0 * single_std

    # ------------------------------------------------------------------
    # Panel A: Single GP
    # ------------------------------------------------------------------
    ax_A.set_title('A — Single GP', fontweight='bold', pad=6)

    # IID vs OOD shading
    ax_A.axvspan(x_min - margin, src_boundary,
                  alpha=0.12, color=config.PALETTE['iid'], zorder=1)
    ax_A.axvspan(src_boundary, x_max + margin,
                  alpha=0.12, color=config.PALETTE['ood'], zorder=1)
    ax_A.axvline(src_boundary, color='#888888', linestyle='--',
                  linewidth=0.9, alpha=0.7)

    ax_A.fill_between(X_grid, single_band_lo, single_band_hi,
                       alpha=0.30, color=config.PALETTE['single'],
                       label='±2σ (single GP)')
    ax_A.plot(X_grid, single_mu, color=config.PALETTE['single'],
               linewidth=2.0, label='Posterior mean')

    # Training data scatter
    rng = np.random.RandomState(config.RANDOM_SEED)
    for h in config.SOURCE_HOSPITALS:
        if h not in features_dict:
            continue
        n_show = min(100, len(features_dict[h]))
        idx    = rng.choice(len(features_dict[h]), size=n_show, replace=False)
        ax_A.scatter(
            features_dict[h][idx, top_feature_idx],
            labels_dict[h][idx].astype(float),
            c=config.PALETTE['hospitals'][h], s=6, alpha=0.4, zorder=4,
        )

    ax_A.set_xlabel(f'PCA-{top_feature_idx} (top expansion feature)')
    ax_A.set_ylabel('P(tumor)')
    ax_A.set_ylim(-0.2, 1.2)
    ax_A.legend(fontsize=7.5, loc='upper left')
    ax_A.text(0.5, 0.05, 'IID', transform=ax_A.transAxes,
               ha='center', fontsize=8, color='#666666')
    ax_A.text(0.82, 0.05, 'OOD', transform=ax_A.transAxes,
               ha='center', fontsize=8, color=config.PALETTE['ood'],
               fontweight='bold')
    _stat_box(ax_A, "Narrow band\nin OOD region")

    # ------------------------------------------------------------------
    # Panel B: Credal GP
    # ------------------------------------------------------------------
    ax_B.set_title('B — Credal GP', fontweight='bold', pad=6)

    ax_B.axvspan(x_min - margin, src_boundary,
                  alpha=0.12, color=config.PALETTE['iid'], zorder=1)
    ax_B.axvspan(src_boundary, x_max + margin,
                  alpha=0.12, color=config.PALETTE['ood'], zorder=1)
    ax_B.axvline(src_boundary, color='#888888', linestyle='--',
                  linewidth=0.9, alpha=0.7)

    # Credal envelope: range of posterior means
    ax_B.fill_between(X_grid, lower_mean, upper_mean,
                       alpha=0.35, color=config.PALETTE['credal'],
                       label='Credal envelope')
    ax_B.plot(X_grid, center_mean, color=config.PALETTE['credal'],
               linewidth=2.0, label='Center mean')

    for h in config.SOURCE_HOSPITALS:
        if h not in features_dict:
            continue
        n_show = min(100, len(features_dict[h]))
        idx    = rng.choice(len(features_dict[h]), size=n_show, replace=False)
        ax_B.scatter(
            features_dict[h][idx, top_feature_idx],
            labels_dict[h][idx].astype(float),
            c=config.PALETTE['hospitals'][h], s=6, alpha=0.4, zorder=4,
        )

    ax_B.set_xlabel(f'PCA-{top_feature_idx} (top expansion feature)')
    ax_B.set_ylabel('P(tumor)')
    ax_B.set_ylim(-0.2, 1.2)
    ax_B.legend(fontsize=7.5, loc='upper left')
    ax_B.text(0.5, 0.05, 'IID', transform=ax_B.transAxes,
               ha='center', fontsize=8, color='#666666')
    ax_B.text(0.82, 0.05, 'OOD', transform=ax_B.transAxes,
               ha='center', fontsize=8, color=config.PALETTE['ood'],
               fontweight='bold')
    _stat_box(ax_B, "Wider band\nin OOD region")

    # ------------------------------------------------------------------
    # Panel C: Width overlay comparison
    # ------------------------------------------------------------------
    ax_C.set_title('C — Uncertainty Width Comparison', fontweight='bold', pad=6)

    ax_C.axvspan(x_min - margin, src_boundary,
                  alpha=0.12, color=config.PALETTE['iid'], zorder=1)
    ax_C.axvspan(src_boundary, x_max + margin,
                  alpha=0.12, color=config.PALETTE['ood'], zorder=1)
    ax_C.axvline(src_boundary, color='#888888', linestyle='--',
                  linewidth=0.9, alpha=0.7)

    # Single GP width (2 * std)
    ax_C.plot(X_grid, 4.0 * single_var, color=config.PALETTE['single'],
               linewidth=2.0, linestyle='--', label='Single GP (4σ²)')
    # Credal width
    ax_C.plot(X_grid, credal_width, color=config.PALETTE['credal'],
               linewidth=2.5, label='Credal width')
    ax_C.fill_between(X_grid, 0, credal_width,
                       alpha=0.20, color=config.PALETTE['credal'])

    ax_C.set_xlabel(f'PCA-{top_feature_idx}')
    ax_C.set_ylabel('Uncertainty width')
    ax_C.legend(fontsize=7.5)

    ood_mean_width  = np.mean(credal_width[X_grid > src_boundary])
    iid_mean_width  = np.mean(credal_width[X_grid <= src_boundary])
    ratio           = ood_mean_width / (iid_mean_width + 1e-8)
    _stat_box(ax_C, f"OOD/IID width ratio\n= {ratio:.2f}×")

    # ------------------------------------------------------------------
    # Panel D: Scatter of credal width vs OOD distance + Pearson r
    # ------------------------------------------------------------------
    ax_D.set_title('D — Credal Width vs OOD Distance', fontweight='bold', pad=6)

    # Get credal width at OOD test points
    ood_h    = config.HELD_OUT_HOSPITAL
    X_ood_1d = features_dict[ood_h][:, top_feature_idx]
    rng2     = np.random.RandomState(config.RANDOM_SEED + 7)
    n_show_d = min(400, len(X_ood_1d))
    idx_d    = rng2.choice(len(X_ood_1d), size=n_show_d, replace=False)
    X_ood_s  = X_ood_1d[idx_d]
    dist_s   = ood_distances[idx_d] if len(ood_distances) > max(idx_d) \
               else np.abs(X_ood_s - src_boundary)

    _, _, _, width_at_ood = credal_gp.predict(X_ood_s)

    r_val, p_val = stats.pearsonr(dist_s, width_at_ood)

    sc = ax_D.scatter(dist_s, width_at_ood,
                       c=config.PALETTE['credal'], alpha=0.40,
                       s=12, edgecolors='none', zorder=3)

    # Regression line
    m, b = np.polyfit(dist_s, width_at_ood, 1)
    x_fit = np.linspace(dist_s.min(), dist_s.max(), 100)
    ax_D.plot(x_fit, m * x_fit + b,
               color=config.PALETTE['single'], linewidth=2.0,
               linestyle='--', zorder=4, label='Linear fit')

    ax_D.set_xlabel('OOD distance from source coverage')
    ax_D.set_ylabel('Credal width W(x*)')
    ax_D.legend(fontsize=7.5)
    p_str = f"{p_val:.2e}" if p_val < 0.001 else f"{p_val:.3f}"
    _stat_box(ax_D, f"Pearson r = {r_val:.3f}\np = {p_str}")

    fig.suptitle(
        'Camelyon17-WILDS: Epistemic Uncertainty — Single GP vs Credal GP',
        fontsize=13, fontweight='bold', y=1.01
    )

    _save_figure(fig, 'figure2_credal_gp')
    return fig


# ---------------------------------------------------------------------------
# Figure 3
# ---------------------------------------------------------------------------
def plot_figure3(
    hospital_reductions: Dict[int, float],
    width_before: np.ndarray,
    width_after: np.ndarray,
    X_plot: np.ndarray,
    metrics_results: Dict[str, Dict[str, float]],
    selected_order: List[int],
    mean_widths: List[float],
) -> plt.Figure:
    """
    Figure 3: Domain selection and metrics comparison.

    Args
    ----
    hospital_reductions : {hospital_id: expected_reduction}.
    width_before        : np.ndarray — credal widths before best hospital added.
    width_after         : np.ndarray — credal widths after best hospital added.
    X_plot              : np.ndarray — x-axis values for width panels.
    metrics_results     : {method_name: {metric: float}}.
    selected_order      : list of int — greedy selection order.
    mean_widths         : list of float — width at each greedy round.

    Returns
    -------
    fig : matplotlib Figure
    """
    _apply_style()
    fig = plt.figure(figsize=(16, 5.5))
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.45,
                            left=0.06, right=0.97, bottom=0.13, top=0.88)

    ax_A = fig.add_subplot(gs[0])
    ax_B = fig.add_subplot(gs[1])
    ax_C = fig.add_subplot(gs[2])
    ax_D = fig.add_subplot(gs[3])

    pal = config.PALETTE

    # ------------------------------------------------------------------
    # Panel A: Bar chart — hospital ranking by expected width reduction
    # ------------------------------------------------------------------
    ax_A.set_title('A — Hospital Ranking\n(expected credal width reduction)',
                   fontweight='bold', pad=4)

    sorted_hosps   = sorted(hospital_reductions.items(),
                            key=lambda x: x[1], reverse=True)
    hosp_ids       = [h for h, _ in sorted_hosps]
    reductions_arr = [r for _, r in sorted_hosps]
    bar_colors_A   = [pal['hospitals'][h] for h in hosp_ids]

    x_pos = np.arange(len(hosp_ids))
    bars  = ax_A.bar(x_pos, reductions_arr,
                      color=bar_colors_A, alpha=0.85,
                      edgecolor='white', linewidth=0.5)

    # Mark best
    ax_A.bar(x_pos[0], reductions_arr[0],
              color=bar_colors_A[0], alpha=1.0,
              edgecolor=pal['credal'], linewidth=2.0)

    ax_A.set_xticks(x_pos)
    ax_A.set_xticklabels(
        [f"H{h}" for h in hosp_ids],
        rotation=0, fontsize=9
    )
    ax_A.set_ylabel('Expected Δ credal width')
    ax_A.axhline(0, color='#888888', linewidth=0.8, linestyle='--')

    best_h   = hosp_ids[0]
    best_red = reductions_arr[0]
    ax_A.text(x_pos[0], best_red + 0.002 * abs(best_red),
              f'Best\n(H{best_h})',
              ha='center', fontsize=8, color=pal['credal'],
              fontweight='bold')

    # Hospital name labels below bars
    for xi, h in zip(x_pos, hosp_ids):
        short_name = config.HOSPITAL_NAMES[h].split('(')[0].strip()
        ax_A.text(xi, -0.08,
                  short_name, ha='center', va='top', fontsize=6.5,
                  rotation=15, color='#444444', transform=ax_A.get_xaxis_transform())

    _stat_box(ax_A, f"Best: H{best_h}\nΔ = {best_red:.4f}")

    # ------------------------------------------------------------------
    # Panel B: Before vs After credal width
    # ------------------------------------------------------------------
    ax_B.set_title(f'B — Width Before/After Adding H{best_h}',
                   fontweight='bold', pad=4)

    ax_B.plot(X_plot, width_before, color=pal['single'],
               linewidth=2.0, linestyle='--', label='Before', alpha=0.9)
    ax_B.fill_between(X_plot, 0, width_before,
                       alpha=0.15, color=pal['single'])

    ax_B.plot(X_plot, width_after, color=pal['credal'],
               linewidth=2.5, label=f'After adding H{best_h}', alpha=0.9)
    ax_B.fill_between(X_plot, 0, width_after,
                       alpha=0.20, color=pal['credal'])

    # Shade the region where width reduced most
    ax_B.fill_between(X_plot,
                       width_after, width_before,
                       where=(width_before > width_after),
                       alpha=0.25, color='#3BB273',
                       label='Width reduction')

    ax_B.set_xlabel(f'PCA feature value')
    ax_B.set_ylabel('Credal width')
    ax_B.legend(fontsize=7.5)

    mean_before = width_before.mean()
    mean_after  = width_after.mean()
    pct_red     = (mean_before - mean_after) / (mean_before + 1e-8) * 100
    _stat_box(ax_B, f"Mean Δ = {mean_before - mean_after:.4f}\n({pct_red:.1f}% reduction)")

    # ------------------------------------------------------------------
    # Panel C: Metrics comparison table (visual)
    # ------------------------------------------------------------------
    ax_C.set_title('C — Metrics Comparison', fontweight='bold', pad=4)
    ax_C.axis('off')

    methods  = list(metrics_results.keys())
    metrics_order = ['ece', 'nll', 'brier', 'auroc', 'fpr95']
    m_labels      = ['ECE↓', 'NLL↓', 'Brier↓', 'AUROC↑', 'FPR@95↓']

    n_rows = len(methods) + 1   # +1 header
    n_cols = len(metrics_order) + 1   # +1 method name column

    table_data = [['Method'] + m_labels]
    for mth in methods:
        row = [mth]
        for m in metrics_order:
            v = metrics_results[mth].get(m, float('nan'))
            row.append(f"{v:.3f}" if not np.isnan(v) else 'N/A')
        table_data.append(row)

    # Determine which cells are best
    best_cells = {}
    for ci, m in enumerate(metrics_order):
        col_vals = []
        for ri, mth in enumerate(methods):
            v = metrics_results[mth].get(m, float('nan'))
            if not np.isnan(v):
                col_vals.append((v, ri))
        if col_vals:
            if m in ('auroc',):
                best_ri = max(col_vals, key=lambda x: x[0])[1]
            else:
                best_ri = min(col_vals, key=lambda x: x[0])[1]
            best_cells[ci] = best_ri + 1   # +1 for header row offset

    table = ax_C.table(
        cellText=table_data,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Style header row
    for j in range(n_cols):
        table[0, j].set_facecolor('#2E4057')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style body rows
    method_colors = {
        'Credal GP':    pal['credal'],
        'ERM':          pal['erm'],
        'MC Dropout':   pal['mc'],
        'Deep Ensemble':pal['ensemble'],
    }
    for ri in range(1, n_rows):
        mth_name = methods[ri - 1]
        row_color = method_colors.get(mth_name, '#FFFFFF')
        # Method name cell
        table[ri, 0].set_facecolor(row_color)
        table[ri, 0].set_text_props(color='white', fontweight='bold',
                                      fontsize=7.5)
        for ci in range(1, n_cols):
            # Check if best
            if best_cells.get(ci - 1) == ri:
                table[ri, ci].set_facecolor('#E8F8E8')
                table[ri, ci].set_text_props(fontweight='bold')
            else:
                table[ri, ci].set_facecolor('#F8F8F8' if ri % 2 == 0
                                             else 'white')

    # ------------------------------------------------------------------
    # Panel D: Greedy selection improvement curve
    # ------------------------------------------------------------------
    ax_D.set_title('D — Greedy Selection Curve', fontweight='bold', pad=4)

    rounds   = list(range(len(mean_widths)))
    x_labels = [f"R{i}\n(H{selected_order[i]})"
                 if i < len(selected_order) else f"R{i}"
                 for i in rounds]

    ax_D.plot(rounds, mean_widths,
               color=pal['credal'], linewidth=2.5,
               marker='o', markersize=7, zorder=4,
               label='Mean credal width')
    ax_D.fill_between(rounds, mean_widths,
                       alpha=0.20, color=pal['credal'])

    # Annotate each point
    for i, (x, y) in enumerate(zip(rounds, mean_widths)):
        lbl = x_labels[i].replace('\n', ' ')
        ax_D.annotate(
            lbl,
            (x, y),
            textcoords='offset points',
            xytext=(0, 8),
            ha='center',
            fontsize=7.5,
            color='#333333',
        )

    ax_D.set_xlabel('Selection round')
    ax_D.set_ylabel('Mean credal width at OOD hospital')
    ax_D.set_xticks(rounds)
    ax_D.set_xticklabels([f"R{i}" for i in rounds], fontsize=8)

    total_red = mean_widths[0] - mean_widths[-1]
    pct_total = total_red / (mean_widths[0] + 1e-8) * 100
    _stat_box(ax_D, f"Total reduction:\n{total_red:.4f} ({pct_total:.1f}%)")

    fig.suptitle(
        'Camelyon17-WILDS: Domain Selection and Method Comparison',
        fontsize=13, fontweight='bold', y=1.01
    )

    _save_figure(fig, 'figure3_domain_selection')
    return fig


# ---------------------------------------------------------------------------
# Convenience: plot all 3 figures
# ---------------------------------------------------------------------------
def plot_all_figures(
    features_dict: Dict[int, np.ndarray],
    labels_dict: Dict[int, np.ndarray],
    expansions: Dict[int, float],
    top_feature_indices: List[int],
    credal_gp,
    top_feature_idx: int,
    ood_distances: np.ndarray,
    hospital_reductions: Dict[int, float],
    width_before: np.ndarray,
    width_after: np.ndarray,
    X_plot: np.ndarray,
    metrics_results: Dict[str, Dict[str, float]],
    selected_order: List[int],
    mean_widths: List[float],
) -> None:
    """
    Generate and save all three publication figures.

    Calls plot_figure1, plot_figure2, plot_figure3 in sequence and closes
    each figure to free memory.
    """
    print("\n  Generating Figure 1: Domain profiles + expansion function …")
    fig1 = plot_figure1(features_dict, expansions, top_feature_indices)
    plt.close(fig1)

    print("  Generating Figure 2: Single GP vs Credal GP …")
    fig2 = plot_figure2(
        features_dict, labels_dict, credal_gp,
        top_feature_idx, ood_distances
    )
    plt.close(fig2)

    print("  Generating Figure 3: Domain selection + metrics …")
    fig3 = plot_figure3(
        hospital_reductions, width_before, width_after, X_plot,
        metrics_results, selected_order, mean_widths
    )
    plt.close(fig3)

    print(f"\n  All figures saved to: {config.OUTPUTS_DIR}")
