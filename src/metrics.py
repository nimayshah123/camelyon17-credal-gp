"""
metrics.py — Evaluation metrics for calibration and OOD detection.

Metrics computed
----------------
ECE   (Expected Calibration Error)  — calibration quality.
NLL   (Negative Log Likelihood)     — probabilistic accuracy.
Brier Score                         — mean squared error of probabilities.
AUROC (Area Under ROC Curve)        — OOD detection performance when
                                       uncertainty score used to separate
                                       IID vs OOD samples.
FPR@95 (False Positive Rate at 95%  — complementary OOD metric.
         True Positive Rate)

All metrics follow sklearn / torchmetrics conventions.
"""

import os
import sys
import numpy as np
from typing import Dict, Optional, List

from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ---------------------------------------------------------------------------
# ECE
# ---------------------------------------------------------------------------
def compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = None,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Partitions [0,1] into n_bins equal-width confidence buckets.
    In each bucket B_m:
        ECE += |B_m|/N * |accuracy(B_m) - confidence(B_m)|

    Args
    ----
    y_true : np.ndarray, shape (N,)  — binary ground truth.
    y_prob : np.ndarray, shape (N,)  — predicted probabilities ∈ [0,1].
    n_bins : int                     — number of calibration bins.

    Returns
    -------
    float : ECE ∈ [0, 1].  Lower is better.
    """
    n_bins = n_bins or config.ECE_BINS
    y_true = np.asarray(y_true).ravel()
    y_prob = np.clip(np.asarray(y_prob).ravel(), 1e-8, 1.0 - 1e-8)
    N      = len(y_true)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece       = 0.0

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_prob >= lo) & (y_prob < hi)
        if i == n_bins - 1:          # include right edge in last bin
            mask = (y_prob >= lo) & (y_prob <= hi)

        n_m = mask.sum()
        if n_m == 0:
            continue

        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (n_m / N) * abs(acc - conf)

    return float(ece)


# ---------------------------------------------------------------------------
# NLL
# ---------------------------------------------------------------------------
def compute_nll(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Compute Binary Negative Log Likelihood (cross-entropy loss).

    NLL = −(1/N) Σ [ y_i log(p_i) + (1−y_i) log(1−p_i) ]

    Args
    ----
    y_true : np.ndarray, shape (N,)
    y_prob : np.ndarray, shape (N,)

    Returns
    -------
    float : NLL ≥ 0.  Lower is better.
    """
    y_true = np.asarray(y_true).ravel().astype(float)
    y_prob = np.nan_to_num(np.asarray(y_prob).ravel(), nan=0.5, posinf=1.0, neginf=0.0)
    y_prob = np.clip(y_prob, 1e-8, 1.0 - 1e-8)

    nll = -(y_true * np.log(y_prob) + (1.0 - y_true) * np.log(1.0 - y_prob))
    return float(nll.mean())


# ---------------------------------------------------------------------------
# Brier Score
# ---------------------------------------------------------------------------
def compute_brier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Compute Brier Score.

    BS = (1/N) Σ (p_i − y_i)²

    Args
    ----
    y_true : np.ndarray, shape (N,)
    y_prob : np.ndarray, shape (N,)

    Returns
    -------
    float : Brier ∈ [0, 1].  Lower is better.
    """
    y_true = np.asarray(y_true).ravel().astype(float)
    y_prob = np.asarray(y_prob).ravel().astype(float)
    return float(np.mean((y_prob - y_true) ** 2))


# ---------------------------------------------------------------------------
# AUROC (OOD detection)
# ---------------------------------------------------------------------------
def compute_auroc(
    is_ood: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> float:
    """
    Compute AUROC for OOD detection using uncertainty as the detector score.

    Convention: higher uncertainty → more likely OOD.
    is_ood = 1 for OOD samples, 0 for IID samples.

    Args
    ----
    is_ood            : np.ndarray, shape (N,) — binary OOD label (0/1).
    uncertainty_scores: np.ndarray, shape (N,) — uncertainty / credal width.

    Returns
    -------
    float : AUROC ∈ [0, 1].  Higher is better for OOD detection.
    """
    is_ood  = np.asarray(is_ood).ravel().astype(int)
    scores  = np.asarray(uncertainty_scores).ravel()

    if len(np.unique(is_ood)) < 2:
        return 0.5   # degenerate case

    return float(roc_auc_score(is_ood, scores))


# ---------------------------------------------------------------------------
# FPR@95
# ---------------------------------------------------------------------------
def compute_fpr95(
    is_ood: np.ndarray,
    uncertainty_scores: np.ndarray,
) -> float:
    """
    Compute False Positive Rate when True Positive Rate = 95%.

    Args
    ----
    is_ood            : np.ndarray, shape (N,) — binary OOD label (0/1).
    uncertainty_scores: np.ndarray, shape (N,) — uncertainty score.

    Returns
    -------
    float : FPR@TPR=95% ∈ [0, 1].  Lower is better.
    """
    is_ood = np.asarray(is_ood).ravel().astype(int)
    scores = np.asarray(uncertainty_scores).ravel()

    if len(np.unique(is_ood)) < 2:
        return 1.0

    fpr, tpr, _ = roc_curve(is_ood, scores)

    # Find the first index where TPR >= 0.95
    indices = np.where(tpr >= 0.95)[0]
    if len(indices) == 0:
        return float(fpr[-1])
    return float(fpr[indices[0]])


# ---------------------------------------------------------------------------
# Composite evaluator
# ---------------------------------------------------------------------------
def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    uncertainty_scores: np.ndarray,
    is_ood: np.ndarray = None,
    n_bins: int = None,
) -> Dict[str, float]:
    """
    Compute all metrics and return as a dict.

    Args
    ----
    y_true            : np.ndarray (N,) — binary labels.
    y_prob            : np.ndarray (N,) — predicted probabilities.
    uncertainty_scores: np.ndarray (N,) — uncertainty / credal width values.
    is_ood            : np.ndarray (N,) — 1 if OOD, 0 if IID.  Optional;
                        if None all samples are treated as IID and
                        AUROC / FPR95 return 0.5 / 1.0 respectively.
    n_bins            : int — ECE bins.

    Returns
    -------
    dict with keys: 'ece', 'nll', 'brier', 'auroc', 'fpr95'
    """
    if is_ood is None:
        is_ood = np.zeros(len(y_true), dtype=int)
    return {
        'ece':   compute_ece(y_true, y_prob, n_bins),
        'nll':   compute_nll(y_true, y_prob),
        'brier': compute_brier(y_true, y_prob),
        'auroc': compute_auroc(is_ood, uncertainty_scores),
        'fpr95': compute_fpr95(is_ood, uncertainty_scores),
    }


def print_metrics_table(results_dict: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted comparison table for all methods.

    Args
    ----
    results_dict : dict
        {'method_name': {'ece': float, 'nll': float, 'brier': float,
                         'auroc': float, 'fpr95': float}, ...}
    """
    header_width = 22
    col_width    = 10

    methods = list(results_dict.keys())
    metrics  = ['ece', 'nll', 'brier', 'auroc', 'fpr95']
    labels   = ['ECE(v)', 'NLL(v)', 'Brier(v)', 'AUROC(^)', 'FPR95(v)']

    # Header
    header = f"{'Method':<{header_width}}"
    for lbl in labels:
        header += f"{lbl:>{col_width}}"
    print()
    print("  " + header)
    print("  " + "-" * (header_width + col_width * len(metrics)))

    # Rows
    for method in methods:
        row = f"{method:<{header_width}}"
        for m in metrics:
            val = results_dict[method].get(m, float('nan'))
            row += f"{val:>{col_width}.4f}"
        print("  " + row)

    print()

    # Highlight best per metric
    best_row = f"{'Best':<{header_width}}"
    for i, m in enumerate(metrics):
        vals = [results_dict[mth].get(m, float('nan')) for mth in methods]
        vals_valid = [(v, mth) for v, mth in zip(vals, methods)
                      if not np.isnan(v)]
        if not vals_valid:
            best_row += f"{'N/A':>{col_width}}"
            continue
        if m in ('auroc',):                # higher is better
            best_val, best_mth = max(vals_valid, key=lambda x: x[0])
        else:                              # lower is better
            best_val, best_mth = min(vals_valid, key=lambda x: x[0])
        best_row += f"{'['+best_mth[:6]+']':>{col_width}}"

    print("  " + best_row)
    print()


# ---------------------------------------------------------------------------
# Helpers for building is_ood label arrays
# ---------------------------------------------------------------------------
def make_ood_labels(
    n_iid: int,
    n_ood: int,
) -> np.ndarray:
    """
    Build a binary is_ood array: 0 for IID samples, 1 for OOD.

    Args
    ----
    n_iid : int
    n_ood : int

    Returns
    -------
    np.ndarray, shape (n_iid + n_ood,)
    """
    return np.concatenate([np.zeros(n_iid, dtype=int),
                           np.ones(n_ood,  dtype=int)])


def aggregate_baseline_metrics(
    baselines_dict: dict,
    features_dict: dict,
    labels_dict: dict,
    credal_gp,
    top_feature_idx: int,
) -> Dict[str, Dict[str, float]]:
    """
    Run all baselines and the credal GP on the OOD hospital and compile
    the full metrics table.

    Args
    ----
    baselines_dict  : {'erm': ERMBaseline, 'mc': ..., 'ensemble': ...}
    features_dict   : {hospital_id: np.ndarray (N, D)}
    labels_dict     : {hospital_id: np.ndarray (N,)}
    credal_gp       : fitted CredalGP instance
    top_feature_idx : int — PCA dim used as credal GP x-axis

    Returns
    -------
    results_dict : {method_name: {metric_name: float}}
    """
    ood_h     = config.HELD_OUT_HOSPITAL
    src_hs    = config.SOURCE_HOSPITALS

    X_ood_full = features_dict[ood_h]
    y_ood      = labels_dict[ood_h]

    # For OOD detection: pool source test samples as IID, ood as OOD
    # Use a subsample of source for speed
    rng = np.random.RandomState(config.RANDOM_SEED)
    src_parts_X, src_parts_y = [], []
    for h in src_hs:
        if h not in features_dict:
            continue
        n_src = min(len(features_dict[h]), len(X_ood_full))
        idx   = rng.choice(len(features_dict[h]), size=n_src, replace=False)
        src_parts_X.append(features_dict[h][idx])
        src_parts_y.append(labels_dict[h][idx])

    X_iid_full = np.concatenate(src_parts_X, axis=0)
    y_iid      = np.concatenate(src_parts_y, axis=0)

    is_ood = make_ood_labels(len(X_iid_full), len(X_ood_full))
    X_all  = np.concatenate([X_iid_full, X_ood_full], axis=0)
    y_all  = np.concatenate([y_iid,      y_ood],      axis=0)

    results = {}

    # -----------------------------------------------------------------------
    # Credal GP
    # -----------------------------------------------------------------------
    X_gp_all = X_all[:, top_feature_idx]
    c_mean, _, _, c_width = credal_gp.predict(X_gp_all)
    # Sigmoid squash for probabilities
    def _sig(x):
        return np.where(x >= 0, 1/(1+np.exp(-x)), np.exp(x)/(1+np.exp(x)))

    gp_proba = _sig(c_mean)
    gp_proba = np.clip(gp_proba, 1e-6, 1.0 - 1e-6)

    results['Credal GP'] = compute_all_metrics(
        y_all, gp_proba, c_width, is_ood
    )

    # -----------------------------------------------------------------------
    # ERM
    # -----------------------------------------------------------------------
    erm = baselines_dict.get('erm')
    if erm is not None:
        erm_proba = erm.predict_proba(X_all)
        # ERM uncertainty = entropy
        erm_unc = -(
            erm_proba * np.log(erm_proba + 1e-8)
            + (1 - erm_proba) * np.log(1 - erm_proba + 1e-8)
        )
        results['ERM'] = compute_all_metrics(
            y_all, erm_proba, erm_unc, is_ood
        )

    # -----------------------------------------------------------------------
    # MC Dropout
    # -----------------------------------------------------------------------
    mc = baselines_dict.get('mc')
    if mc is not None:
        mc_mean, mc_unc = mc.predict_proba(X_all)
        results['MC Dropout'] = compute_all_metrics(
            y_all, mc_mean, mc_unc, is_ood
        )

    # -----------------------------------------------------------------------
    # Deep Ensemble
    # -----------------------------------------------------------------------
    ens = baselines_dict.get('ensemble')
    if ens is not None:
        ens_mean, ens_unc = ens.predict_proba(X_all)
        ens_mean = np.nan_to_num(ens_mean, nan=0.5, posinf=1.0, neginf=0.0)
        ens_unc  = np.nan_to_num(ens_unc,  nan=0.0, posinf=1.0, neginf=0.0)
        results['Deep Ensemble'] = compute_all_metrics(
            y_all, ens_mean, ens_unc, is_ood
        )

    return results
