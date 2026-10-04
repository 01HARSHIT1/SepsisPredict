"""Shared training and inference boundary.

FastAPI will import this package later. The frontend will not reimplement it.
Phase 1 defines the contract and the feature transform. It does not serve predictions.
"""

from src.inference.service import ModelNotTrainedError, require_trained_model

__all__ = ["ModelNotTrainedError", "require_trained_model"]
