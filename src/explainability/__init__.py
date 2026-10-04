"""Explainability is not computed in phase 1.

After an XGBoost model is trained, SHAP values from that model are the planned
source of feature attributions. Explanations must not be written by hand from
clinical thresholds.
"""
