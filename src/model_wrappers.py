"""Small prediction wrappers shared by training and inference."""
from __future__ import annotations

import numpy as np


def is_model_sequence(value) -> bool:
    """Return True for list/tuple model ensembles, but not estimator objects."""
    return isinstance(value, (list, tuple))


def get_model_feature_names(model) -> list[str] | None:
    """Return persisted feature names for sklearn/CatBoost-style estimators."""
    if is_model_sequence(model):
        if not model:
            return None
        return get_model_feature_names(model[0])
    names = getattr(model, "feature_name_", None)
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    if names is None:
        names = getattr(model, "feature_names_", None)
    return list(names) if names is not None else None


def predict_model_or_ensemble(model_or_models, X):
    """Predict with either one fitted model, a list of models, or a wrapper."""
    if is_model_sequence(model_or_models):
        if not model_or_models:
            raise ValueError("Cannot predict from an empty model ensemble.")
        preds = [model.predict(X) for model in model_or_models]
        return np.mean(np.vstack(preds), axis=0)
    return model_or_models.predict(X)


class AveragingRegressor:
    """Average predictions from several fitted regressors.

    This keeps fold ensembles compatible with the existing ``models[week].predict``
    inference path while preserving feature-name metadata for ``predict.py``.
    """

    def __init__(self, models: list, feature_names: list[str] | None = None):
        if not models:
            raise ValueError("AveragingRegressor requires at least one model.")
        self.models = list(models)
        self.feature_name_ = feature_names or get_model_feature_names(self.models[0])
        if self.feature_name_ is not None:
            self.feature_names_in_ = np.array(self.feature_name_, dtype=object)

    def predict(self, X):
        return predict_model_or_ensemble(self.models, X)

    def __len__(self) -> int:
        return len(self.models)
