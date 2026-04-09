"""
domain_selection.py — Greedy hospital ranking by expected credal width reduction.

Closed-loop active domain selection
------------------------------------
The expansion function identifies *where* the credal GP is widest (high
epistemic uncertainty).  The domain selection step answers: which hospital
should we collect data from next to most efficiently reduce that uncertainty?

Algorithm
---------
For each candidate hospital h ∈ {0, 1, 2, 3, 4}:
  1. Compute the current credal width W_0(x) at query points X_q.
  2. Hypothetically add hospital h's data to the training set.
  3. Compute the resulting credal width W_h(x) after augmentation.
  4. Compute expected reduction: Δ_h = mean(W_0 - W_h) at X_q.
  5. Rank hospitals by Δ_h descending.

For greedy multi-round selection:
  Round 1: rank all hospitals, add the best one.
  Round 2: repeat on the remaining hospitals, with the round-1 hospital
           already included in the training set.
  … and so on.

Functions
---------
compute_expected_width_reduction    : Δ for one candidate hospital.
rank_hospitals_by_reduction         : full ranking dict.
greedy_hospital_selection           : multi-round greedy selection curve.
"""

import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.credal_gp import CredalGP


def compute_expected_width_reduction(
    gp: CredalGP,
    candidate_features: np.ndarray,
    candidate_labels: np.ndarray,
    X_query: np.ndarray,
) -> float:
    """
    Compute expected credal width reduction from adding one candidate hospital.

    Args
    ----
    gp                : fitted CredalGP (trained on current source hospitals).
    candidate_features: np.ndarray, shape (N_cand,) — 1-D feature values of
                        the candidate hospital (top-expansion feature).
    candidate_labels  : np.ndarray, shape (N_cand,) — labels.
    X_query           : np.ndarray, shape (M,) — query points (typically the
                        OOD hospital's feature values).

    Returns
    -------
    float : mean(W_current − W_after_adding_candidate).  Positive = reduction.
    """
    # Current width at query points
    _, _, _, W_current = gp.predict(X_query)

    # Width after hypothetically adding candidate hospital
    W_after = gp.expected_width_with_new_domain(
        candidate_features, candidate_labels, X_query
    )

    reduction = np.mean(W_current - W_after)
    return float(reduction)


def rank_hospitals_by_reduction(
    gp: CredalGP,
    features_dict: Dict[int, np.ndarray],
    labels_dict: Dict[int, np.ndarray],
    top_feature_idx: int,
    query_hospital: int = None,
    candidate_hospitals: List[int] = None,
    subsample_cand: int = 200,
    subsample_query: int = 300,
) -> Dict[int, float]:
    """
    Rank all candidate hospitals by their expected credal width reduction.

    Args
    ----
    gp               : fitted CredalGP.
    features_dict    : {hospital_id: np.ndarray (N, D)}.
    labels_dict      : {hospital_id: np.ndarray (N,)}.
    top_feature_idx  : int — which PCA dimension is the GP x-axis.
    query_hospital   : int — hospital whose region we want to cover.
                       Defaults to config.HELD_OUT_HOSPITAL.
    candidate_hospitals : list of int — hospitals to rank.
                          Defaults to all 5 hospitals.
    subsample_cand   : int — max samples from each candidate hospital.
    subsample_query  : int — max query points from the query hospital.

    Returns
    -------
    dict : {hospital_id: expected_reduction_float}, sorted by reduction desc.
    """
    query_hospital      = query_hospital if query_hospital is not None \
                         else config.HELD_OUT_HOSPITAL
    candidate_hospitals = candidate_hospitals if candidate_hospitals is not None \
                         else list(range(config.N_HOSPITALS))

    rng = np.random.RandomState(config.RANDOM_SEED)

    # Query points: feature values of the held-out OOD hospital
    q_feats = features_dict[query_hospital][:, top_feature_idx]
    if len(q_feats) > subsample_query:
        idx    = rng.choice(len(q_feats), size=subsample_query, replace=False)
        q_feats = q_feats[idx]
    X_query = q_feats

    reductions = {}
    for h in candidate_hospitals:
        if h not in features_dict:
            reductions[h] = 0.0
            continue

        cand_feats = features_dict[h][:, top_feature_idx]
        cand_y     = labels_dict[h].astype(float)

        if len(cand_feats) > subsample_cand:
            idx        = rng.choice(len(cand_feats), size=subsample_cand,
                                    replace=False)
            cand_feats = cand_feats[idx]
            cand_y     = cand_y[idx]

        delta = compute_expected_width_reduction(
            gp, cand_feats, cand_y, X_query
        )
        reductions[h] = delta

    # Sort descending by reduction
    sorted_reductions = dict(
        sorted(reductions.items(), key=lambda x: x[1], reverse=True)
    )
    return sorted_reductions


def greedy_hospital_selection(
    features_dict: Dict[int, np.ndarray],
    labels_dict: Dict[int, np.ndarray],
    top_feature_idx: int,
    n_rounds: int = None,
    query_hospital: int = None,
    verbose: bool = True,
) -> Tuple[List[int], List[float], List[float]]:
    """
    Greedy multi-round hospital selection.

    In each round:
      - Fit a credal GP on all currently selected hospitals.
      - Rank remaining hospitals by expected credal width reduction.
      - Add the best hospital to the selected set.

    Args
    ----
    features_dict   : {hospital_id: np.ndarray (N, D)}.
    labels_dict     : {hospital_id: np.ndarray (N,)}.
    top_feature_idx : int — which PCA dimension is the GP x-axis.
    n_rounds        : int — how many hospitals to greedily add.
                      Defaults to N_HOSPITALS − 1 (all but the query).
    query_hospital  : int — the target OOD hospital.
                      Defaults to config.HELD_OUT_HOSPITAL.
    verbose         : bool

    Returns
    -------
    selected_order : list of int — hospitals added in order.
    mean_widths    : list of float — mean credal width after each addition.
    round_reductions: list of float — Δ at each round.
    """
    query_hospital = query_hospital if query_hospital is not None \
                     else config.HELD_OUT_HOSPITAL
    n_rounds       = n_rounds or (config.N_HOSPITALS - 1)

    all_hospitals   = list(range(config.N_HOSPITALS))
    remaining       = [h for h in all_hospitals if h != query_hospital]
    selected        = []

    selected_order   = []
    mean_widths      = []
    round_reductions = []

    rng = np.random.RandomState(config.RANDOM_SEED)

    # Compute initial credal width (no hospitals selected — use empty GP
    # or a minimal starting set with one hospital)
    # Start with the hospital that has the most samples
    first_h = max(remaining, key=lambda h: len(features_dict.get(h, [])))
    selected.append(first_h)
    remaining.remove(first_h)
    selected_order.append(first_h)

    if verbose:
        print(f"  Round 0: starting with hospital {first_h} "
              f"({config.HOSPITAL_NAMES[first_h]})")

    for round_idx in range(1, n_rounds):
        if not remaining:
            break

        # Fit GP on current selected set
        X_parts, y_parts = [], []
        for h in selected:
            X_parts.append(features_dict[h][:, top_feature_idx])
            y_parts.append(labels_dict[h].astype(float))
        X_train_gp = np.concatenate(X_parts)
        y_train_gp = np.concatenate(y_parts)

        gp = CredalGP()
        gp.fit(X_train_gp, y_train_gp)

        # Query points: OOD hospital
        q_feats = features_dict[query_hospital][:, top_feature_idx]
        n_q     = min(len(q_feats), 300)
        idx     = rng.choice(len(q_feats), size=n_q, replace=False)
        X_query = q_feats[idx]

        _, _, _, W_current = gp.predict(X_query)
        mean_widths.append(float(W_current.mean()))

        # Rank remaining hospitals
        reductions = rank_hospitals_by_reduction(
            gp, features_dict, labels_dict,
            top_feature_idx=top_feature_idx,
            query_hospital=query_hospital,
            candidate_hospitals=remaining,
        )

        best_h = list(reductions.keys())[0]
        best_delta = reductions[best_h]

        selected.append(best_h)
        remaining.remove(best_h)
        selected_order.append(best_h)
        round_reductions.append(best_delta)

        if verbose:
            print(f"  Round {round_idx}: add hospital {best_h} "
                  f"({config.HOSPITAL_NAMES[best_h]}), "
                  f"Δ = {best_delta:.4f}, "
                  f"mean_width = {mean_widths[-1]:.4f}")

    # Final width after all selected
    X_parts, y_parts = [], []
    for h in selected:
        X_parts.append(features_dict[h][:, top_feature_idx])
        y_parts.append(labels_dict[h].astype(float))
    X_train_gp = np.concatenate(X_parts)
    y_train_gp = np.concatenate(y_parts)

    gp_final = CredalGP()
    gp_final.fit(X_train_gp, y_train_gp)

    q_feats = features_dict[query_hospital][:, top_feature_idx]
    n_q     = min(len(q_feats), 300)
    idx     = rng.choice(len(q_feats), size=n_q, replace=False)
    X_query = q_feats[idx]

    _, _, _, W_final = gp_final.predict(X_query)
    mean_widths.append(float(W_final.mean()))

    if verbose:
        print(f"\n  Final mean credal width: {mean_widths[-1]:.4f} "
              f"(started at {mean_widths[0]:.4f})")
        print(f"  Hospital selection order: "
              + " → ".join(str(h) for h in selected_order))

    return selected_order, mean_widths, round_reductions


def compute_before_after_widths(
    gp_before: CredalGP,
    features_dict: Dict[int, np.ndarray],
    labels_dict: Dict[int, np.ndarray],
    top_feature_idx: int,
    best_hospital: int,
    X_plot: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute credal widths before and after adding the best hospital.

    Args
    ----
    gp_before        : fitted CredalGP on source hospitals (without best_h).
    features_dict    : {hospital_id: np.ndarray (N, D)}.
    labels_dict      : {hospital_id: np.ndarray (N,)}.
    top_feature_idx  : int
    best_hospital    : int — the hospital to add.
    X_plot           : np.ndarray, shape (M,) — grid to evaluate widths on.

    Returns
    -------
    (width_before, width_after) : two np.ndarray shape (M,)
    """
    _, _, _, width_before = gp_before.predict(X_plot)

    X_new = features_dict[best_hospital][:, top_feature_idx]
    y_new = labels_dict[best_hospital].astype(float)

    width_after = gp_before.expected_width_with_new_domain(
        X_new, y_new, X_plot
    )
    return width_before, width_after
