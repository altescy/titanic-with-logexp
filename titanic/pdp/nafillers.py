from __future__ import annotations
import typing as tp

import colt
import pandas as pd
import pdpipe as pdp
from pdpipe.col_generation import ColumnTransformer


@colt.register("pdp:fill_na")
class FillNa(ColumnTransformer):
    FILL_TYPES = ["mean", "median", "replace", "mode"]

    def __init__(
            self,
            columns: tp.Union[str, tp.List[str]],
            fill_type: str,
            value: tp.Any = None,
            **kwargs,
    ) -> None:
        super().__init__(columns, **kwargs)
        assert fill_type in FillNa.FILL_TYPES

        self._fill_type = fill_type
        self._value = value
        self._fitted_values: tp.Dict[str, tp.Any] = {}

    def _compute_value(self, series):
        if self._fill_type == "mean":
            return series.dropna().mean()

        if self._fill_type == "median":
            return series.dropna().median()

        if self._fill_type == "replace":
            return self._value

        if self._fill_type == "mode":
            return series.dropna().mode()[0]

        raise RuntimeError(f"not supported fill_type: {self._fill_type}")

    def _fit_transform(self, df, verbose):
        for col in self._get_columns(df):
            self._fitted_values[col] = self._compute_value(df[col])
        return super()._fit_transform(df, verbose)

    def _col_transform(self, series, label):
        if not self._fitted_values:
            value = self._compute_value(series)
        else:
            value = self._fitted_values[series.name]
        return series.fillna(value)
