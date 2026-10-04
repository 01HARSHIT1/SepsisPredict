"""Training code lives here. Scoring does not.

A trained estimator is loaded only through ``src.inference.service``. FastAPI
and the frontend depend on that service and on ``src.inference.contract``,
not on a particular library. Phase 1 has not fit an estimator.



No estimator is fit here. The planned sequence, after the audit, is:

A. Logistic regression on a small current-value plus missingness design, as an
   interpretable check of preprocessing and leakage.
B. Gradient-boosted trees using current values, missing indicators, last
   observations, time since measurement, selected backward features, and
   static context.
C. A GRU or LSTM over the hourly state, including sparse-lab representation.
D. A temporal attention model only if A-C have been trained and compared.

Calibration, if used, is fit on the validation patients only.
"""
