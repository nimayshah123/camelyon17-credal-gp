"""
src — Source modules for the Camelyon17 Credal GP pipeline.

Modules
-------
data_loader       : Load Camelyon17-WILDS and split by hospital.
feature_extractor : ResNet50 embedding extraction and PCA reduction.
expansion         : Expansion function e_φ(z) = σ²_between / σ²_within.
credal_gp         : Credal GP with diverse kernel set.
baselines         : ERM, MC Dropout, Deep Ensemble baselines.
metrics           : ECE, NLL, Brier, AUROC, FPR@95.
domain_selection  : Greedy hospital ranking by credal width reduction.
visualize         : Publication-quality figures (Figures 1–3).
"""
