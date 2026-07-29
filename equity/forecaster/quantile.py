"""
@module: equity.forecaster.quantile
@depends: numpy, pandas, lightgbm
@exports: QuantileHeadForecaster
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7, Q7 (90% interval)
@data_flow: features + ret_hN -> LightGBM quantile (alpha∈{0.05,0.5,0.95}) per horizon
            -> pred_hN_q{05,50,95}

S5.4 quantile heads. Instantiates ``lgb.LGBMRegressor(objective='quantile',
alpha=...)`` directly (the SCE ``build_model('lightgbm')`` factory does not
expose quantile alpha). 15 models = 5 horizons × 3 quantiles.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from equity.forecaster._splits import make_design_matrix, rolling_ts_folds, ts_group_split
from equity.forecaster.config import (
    DEFAULT_HORIZONS,
    DEFAULT_QUANTILES,
    HorizonConfig,
    QuantileHeadParams,
)
from equity.forecaster.targets import forward_target_col


def _quantile_col(horizon: int, q: float) -> str:
    """``pred_h5_q05`` / ``pred_h5_q50`` / ``pred_h5_q95`` naming."""
    pct = int(round(float(q) * 100))
    return f"pred_h{int(horizon)}_q{pct:02d}"


def _new_lgbm(params: QuantileHeadParams, *, seed: int, alpha: float):
    try:
        from lightgbm import LGBMRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lightgbm is required for QuantileHeadForecaster "
            "(install the [models] extra)"
        ) from exc
    return LGBMRegressor(**params.to_lgbm_kwargs(seed=seed, alpha=alpha))


class QuantileHeadForecaster:
    """Per-(horizon, quantile) LightGBM heads → ``pred_hN_q{05,50,95}``."""

    def __init__(
        self,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        quantiles: Sequence[float] = DEFAULT_QUANTILES,
        params: Optional[QuantileHeadParams] = None,
        horizon_cfg: Optional[HorizonConfig] = None,
        feature_cols: Optional[Sequence[str]] = None,
        n_folds: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        cfg = horizon_cfg or HorizonConfig()
        self.horizons: tuple[int, ...] = tuple(int(h) for h in (horizons or cfg.horizons))
        self.quantiles: tuple[float, ...] = tuple(
            float(q) for q in (quantiles or cfg.quantiles)
        )
        self.params = params or QuantileHeadParams()
        self.time_col = cfg.time_col
        self.sector_col = cfg.sector_col
        self.n_folds = int(n_folds if n_folds is not None else cfg.n_folds)
        self.seed = int(seed if seed is not None else cfg.seed)
        self.feature_cols_arg = list(feature_cols) if feature_cols is not None else None

        self.models_: Dict[Tuple[int, float], object] = {}
        self.feature_cols_: List[str] = []
        self.oof_: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        return_oof: bool = True,
    ) -> "QuantileHeadForecaster":
        if frame.empty:
            raise ValueError("QuantileHeadForecaster.fit: empty frame")

        sort_keys = [self.time_col] + (["ticker"] if "ticker" in frame.columns else [])
        ordered = frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        _, cols = make_design_matrix(
            ordered,
            feature_cols=self.feature_cols_arg,
            sector_col=self.sector_col,
            cast_sector_category=True,
        )
        # LightGBM wants categorical as 'category' dtype OR integer codes; we
        # cast sector to category and pass categorical_feature by name.
        self.feature_cols_ = cols

        folds = rolling_ts_folds(
            ordered, time_col=self.time_col, n_folds=self.n_folds
        )

        oof_data: Dict[str, np.ndarray] = {}
        for h in self.horizons:
            for q in self.quantiles:
                oof_data[_quantile_col(h, q)] = np.full(len(ordered), np.nan, dtype=float)

        cat_features = [self.sector_col] if self.sector_col in self.feature_cols_ else "auto"

        for h in self.horizons:
            y_col = forward_target_col(h)
            if y_col not in ordered.columns:
                raise ValueError(
                    f"QuantileHeadForecaster: missing target {y_col!r}"
                )

            for q in self.quantiles:
                qcol = _quantile_col(h, q)
                seed_q = self.seed + int(h * 100) + int(round(q * 100))

                for train_idx, val_idx in folds:
                    y_train = ordered.iloc[train_idx][y_col]
                    tr_mask = y_train.notna().to_numpy()
                    if tr_mask.sum() < 2:
                        continue
                    tr_pos = train_idx[tr_mask]
                    X_tr, _ = make_design_matrix(
                        ordered.iloc[tr_pos],
                        feature_cols=self.feature_cols_,
                        sector_col=self.sector_col,
                    )
                    y_tr = ordered.iloc[tr_pos][y_col].astype(float).to_numpy()
                    model = _new_lgbm(self.params, seed=seed_q, alpha=q)
                    model.fit(
                        X_tr,
                        y_tr,
                        categorical_feature=cat_features,
                    )

                    y_val = ordered.iloc[val_idx][y_col]
                    va_mask = y_val.notna().to_numpy()
                    if va_mask.sum() == 0:
                        continue
                    va_pos = val_idx[va_mask]
                    X_va, _ = make_design_matrix(
                        ordered.iloc[va_pos],
                        feature_cols=self.feature_cols_,
                        sector_col=self.sector_col,
                    )
                    if self.sector_col in X_tr.columns and self.sector_col in X_va.columns:
                        X_va[self.sector_col] = pd.Categorical(
                            X_va[self.sector_col],
                            categories=list(X_tr[self.sector_col].cat.categories),
                        )
                    oof_data[qcol][va_pos] = model.predict(X_va)

                # Earliest train-only block stays NaN (no future-trained filler).
                labeled = ordered[y_col].notna().to_numpy()

                # Final full-fit.
                all_lab = np.where(labeled)[0]
                if len(all_lab) < 2:
                    raise ValueError(
                        f"QuantileHeadForecaster: horizon h={h} has <2 labeled rows"
                    )
                X_all, _ = make_design_matrix(
                    ordered.iloc[all_lab],
                    feature_cols=self.feature_cols_,
                    sector_col=self.sector_col,
                )
                y_all = ordered.iloc[all_lab][y_col].astype(float).to_numpy()
                final = _new_lgbm(self.params, seed=seed_q, alpha=q)
                final.fit(X_all, y_all, categorical_feature=cat_features)
                self.models_[(h, q)] = final

        if return_oof:
            keep = [c for c in (self.time_col, "ticker") if c in ordered.columns]
            oof = ordered[keep].copy()
            for col, arr in oof_data.items():
                oof[col] = arr
            self.oof_ = oof
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("QuantileHeadForecaster.predict: call fit() first")
        ordered = frame.reset_index(drop=True)
        out = pd.DataFrame(index=ordered.index)
        if self.time_col in ordered.columns:
            out[self.time_col] = ordered[self.time_col].to_numpy()
        if "ticker" in ordered.columns:
            out["ticker"] = ordered["ticker"].to_numpy()
        X, _ = make_design_matrix(
            ordered,
            feature_cols=self.feature_cols_,
            sector_col=self.sector_col,
        )
        if self.sector_col in X.columns:
            X[self.sector_col] = X[self.sector_col].astype("category")
        for (h, q), model in self.models_.items():
            out[_quantile_col(h, q)] = model.predict(X)
        return out

    def oof_predictions(self) -> pd.DataFrame:
        if self.oof_ is None:
            raise RuntimeError("QuantileHeadForecaster.oof_predictions: no OOF")
        return self.oof_.copy()

    @staticmethod
    def empirical_coverage(
        y_true: pd.Series | np.ndarray,
        q_low: pd.Series | np.ndarray,
        q_high: pd.Series | np.ndarray,
    ) -> float:
        """Fraction of ``y_true`` falling inside ``[q_low, q_high]`` (finite rows)."""
        yt = np.asarray(y_true, dtype=float)
        lo = np.asarray(q_low, dtype=float)
        hi = np.asarray(q_high, dtype=float)
        mask = np.isfinite(yt) & np.isfinite(lo) & np.isfinite(hi)
        if mask.sum() == 0:
            return float("nan")
        inside = (yt[mask] >= lo[mask]) & (yt[mask] <= hi[mask])
        return float(inside.mean())

    def heldout_coverage(
        self,
        frame: pd.DataFrame,
        *,
        test_frac: float = 0.25,
        lo_q: float = 0.05,
        hi_q: float = 0.95,
    ) -> Dict[int, float]:
        """Fit-free helper: ts-group split, fit on train, coverage on test per h.

        Used by tests / smoke to report empirical 90% coverage on a held-out
        fold. Strict ``[0.85, 0.95]`` calibration is deferred to S6 on real
        equity data. Refits models on the train slice only (does not mutate
        ``self``).
        """
        train, test = ts_group_split(
            frame, time_col=self.time_col, test_frac=test_frac
        )
        runner = QuantileHeadForecaster(
            horizons=self.horizons,
            quantiles=self.quantiles,
            params=self.params,
            horizon_cfg=HorizonConfig(
                horizons=self.horizons,
                quantiles=self.quantiles,
                time_col=self.time_col,
                sector_col=self.sector_col,
                n_folds=self.n_folds,
                seed=self.seed,
            ),
            feature_cols=self.feature_cols_arg,
            n_folds=self.n_folds,
            seed=self.seed,
        )
        runner.fit(train, return_oof=False)
        preds = runner.predict(test)
        out: Dict[int, float] = {}
        for h in self.horizons:
            y_col = forward_target_col(h)
            if y_col not in test.columns:
                out[h] = float("nan")
                continue
            lo_c = _quantile_col(h, lo_q)
            hi_c = _quantile_col(h, hi_q)
            y = test[y_col].reset_index(drop=True)
            out[h] = self.empirical_coverage(y, preds[lo_c], preds[hi_c])
        return out
