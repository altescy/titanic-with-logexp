from __future__ import annotations
import typing as tp

import importlib

import colt
from sklearn.base import BaseEstimator


@colt.register("sklearn_model", constructor="from_dict")
class SklearnModelWrapper:
    @classmethod
    def from_dict(cls, model_dict: tp.Dict[str, tp.Any]) -> BaseEstimator:
        module_name = model_dict.pop("@module", "sklearn")
        model_path = model_dict.pop("@model")
        model_path = f"{module_name}.{model_path}"

        module_path, model_name = model_path.rsplit(".", 1)

        module = importlib.import_module(module_path)
        model_cls = getattr(module, model_name)

        if not issubclass(model_cls, BaseEstimator):
            raise ValueError(f"{model_path} is not an estimator")

        return model_cls(**model_dict)
