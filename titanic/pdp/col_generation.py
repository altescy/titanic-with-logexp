from __future__ import annotations
import typing as tp

import colt
import numpy as np
import pandas as pd
import pdpipe as pdp
from pdpipe.col_generation import ColumnTransformer

@colt.register("pdp:cut")
class Cut(ColumnTransformer):
    def __init__(
            self,
            bins: tp.Union[int, tp.List[int]],
            right: bool = True,
            labels: tp.List[str] = None,
            precision: int = 3,
            include_lowest: bool = False,
            duplicates: str = "raise",
            as_str: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._bins = bins
        self._right = right
        self._labels = labels
        self._precision = precision
        self._include_lowest = include_lowest
        self._duplicate = duplicates
        self._bin_dict: tp.Dict[str, np.ndarray] = {}
        self._as_str = as_str

    def _compute_bins(self, series):
        _, bins = pd.cut(
            x=series,
            bins=self._bins,
            right=self._right,
            labels=self._labels,
            retbins=True,
            precision=self._precision,
            duplicates=self._duplicate,
        )
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins

    def _fit_transform(self, df, verbose):
        for col in self._get_columns(df):
            self._bin_dict[col] = self._compute_bins(df[col])
        return super()._fit_transform(df, verbose)

    def _col_transform(self, series, label):
        if not self._bin_dict:
            bins = self._compute_bins(series)
        else:
            bins = self._bin_dict[series.name]
        cut_series = pd.cut(
            x=series,
            bins=bins,
            labels=self._labels,
            precision=self._precision,
            duplicates=self._duplicate
        )
        if self._as_str:
            cut_series = cut_series.astype(str)
        return cut_series


@colt.register("pdp:qcut")
class Qcut(ColumnTransformer):
    def __init__(
            self,
            q: int,
            labels: tp.List[str] = None,
            precision: int = 3,
            duplicates: str = "raise",
            as_str: bool = False,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._q = q
        self._labels = labels
        self._precision = precision
        self._duplicate = duplicates
        self._bin_dict: tp.Dict[str, np.ndarray] = {}
        self._as_str = as_str

    def _compute_bins(self, series):
        _, bins = pd.qcut(
            x=series,
            q=self._q,
            labels=self._labels,
            retbins=True,
            precision=self._precision,
            duplicates=self._duplicate,
        )
        bins[0] = -np.inf
        bins[-1] = np.inf
        return bins

    def _fit_transform(self, df, verbose):
        for col in self._get_columns(df):
            self._bin_dict[col] = self._compute_bins(df[col])
        return super()._fit_transform(df, verbose)

    def _col_transform(self, series, label):
        if not self._bin_dict:
            bins = self._compute_bins(series)
        else:
            bins = self._bin_dict[series.name]
        qcut_series = pd.cut(
            x=series,
            bins=bins,
            labels=self._labels,
            precision=self._precision,
            duplicates=self._duplicate
        )
        if self._as_str:
            qcut_series = qcut_series.astype(str)
        return qcut_series
