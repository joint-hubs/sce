"""
@module: equity.forecaster.residual
@depends: numpy, pandas, xgboost
@exports: InstrumentResidualForecaster
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7.2 (layer 2), §9.9
@data_flow: features + OOF pred_sector_hN + ret_hN
            -> resid_hN = ret_hN - pred_sector_hN (OOF only)
            -> XGBoost residual head
            -> pred_hN = pred_sector_hN + pred_resid_hN

S5.2 instrument-residual forecaster. Residual labels are built exclusively from
OOF sector-head predictions (PRD §9.9). The only prediction-derived feature
allowed in the residual design matrix is ``pred_sector_hN`` (OOF), with train-fold
mean imputation for NaN.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from equity.forecaster._splits import make_design_matrix, rolling_ts_folds
from equity.forecaster.config import DEFAULT_HORIZONS, HorizonConfig, ResidualHeadParams
from equity.forecaster.targets import forward_target_col


def _new_xgb(params: ResidualHeadParams, *, seed: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xgboost is required for InstrumentResidualForecaster") from exc
    return XGBRegressor(**params.to_xgb_kwargs(seed=seed))


def _fill_with_train_mean(
    train_series: pd.Series, test_series: pd.Series
) -> tuple[pd.Series, pd.Series, float]:
    """Impute NaN in the injected pred_sector feature using train-fold mean only."""
    mean = float(train_series.mean(skipna=True)) if train_series.notna().any() else 0.0
    if not np.isfinite(mean):
        mean = 0.0
    tr = train_series.fillna(mean)
    te = test_series.fillna(mean)
    return tr, te, mean


class InstrumentResidualForecaster:
    """Per-horizon residual XGBoost: ``pred_hN = pred_sector_hN + pred_resid_hN``."""

    def __init__(
        self,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        params: Optional[ResidualHeadParams] = None,
        horizon_cfg: Optional[HorizonConfig] = None,
        feature_cols: Optional[Sequence[str]] = None,
        n_folds: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        cfg = horizon_cfg or HorizonConfig()
        self.horizons: tuple[int, ...] = tuple(int(h) for h in (horizons or cfg.horizons))
        self.params = params or ResidualHeadParams()
        self.time_col = cfg.time_col
        self.sector_col = cfg.sector_col
        self.n_folds = int(n_folds if n_folds is not None else cfg.n_folds)
        self.seed = int(seed if seed is not None else cfg.seed)
        self.feature_cols_arg = list(feature_cols) if feature_cols is not None else None

        self.models_: Dict[int, object] = {}
        self.feature_cols_: List[str] = []
        self.pred_sector_means_: Dict[int, float] = {}
        self.oof_: Optional[pd.DataFrame] = None
        self._fitted = False

    def fit(
        self,
        frame: pd.DataFrame,
        *,
        sector_preds: Optional[pd.DataFrame] = None,
        return_oof: bool = True,
    ) -> "InstrumentResidualForecaster":
        """Fit residual heads.

        Parameters
        ----------
        frame:
            Feature panel carrying ``ret_hN`` labels and base features. May also
            carry OOF ``pred_sector_hN`` columns already.
        sector_preds:
            Optional frame with ``pred_sector_hN`` columns aligned to ``frame``
            (same length / row order after time-sort). When provided, these
            **override** any on-frame ``pred_sector_hN`` columns — use this to
            inject the OOF output of :class:`SectorHeadForecaster`.
        """
        if frame.empty:
            raise ValueError("InstrumentResidualForecaster.fit: empty frame")

        sort_keys = [self.time_col] + (["ticker"] if "ticker" in frame.columns else [])
        ordered = frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True).copy()
        if sector_preds is not None:
            sp = sector_preds.copy()
            # Prefer key-based align (ticker + time) so equal-timestamp
            # cross-sections cannot silently permute under two independent sorts.
            join_keys = [
                k for k in (self.time_col, "ticker") if k in ordered.columns and k in sp.columns
            ]
            ps_cols = [f"pred_sector_h{h}" for h in self.horizons]
            missing_ps = [c for c in ps_cols if c not in sp.columns]
            if missing_ps:
                raise ValueError(
                    f"InstrumentResidualForecaster: sector_preds missing {missing_ps}"
                )
            if join_keys:
                drop_existing = [c for c in ps_cols if c in ordered.columns]
                if drop_existing:
                    ordered = ordered.drop(columns=drop_existing)
                merged = ordered.merge(
                    sp[join_keys + ps_cols].drop_duplicates(join_keys),
                    on=join_keys,
                    how="left",
                    sort=False,
                )
                if len(merged) != len(ordered):
                    raise ValueError(
                        "InstrumentResidualForecaster.fit: sector_preds key-merge "
                        "changed row count (duplicate keys in sector_preds?)."
                    )
                ordered = merged
            else:
                sp = sp.reset_index(drop=True)
                if len(sp) != len(ordered):
                    raise ValueError(
                        "InstrumentResidualForecaster.fit: sector_preds length "
                        f"{len(sp)} != frame length {len(ordered)} (after time-sort). "
                        "Pass sector OOF already aligned to a time-sorted frame."
                    )
                for col in ps_cols:
                    ordered[col] = sp[col].to_numpy()

        for h in self.horizons:
            col = f"pred_sector_h{h}"
            if col not in ordered.columns:
                raise ValueError(
                    f"InstrumentResidualForecaster: missing OOF column {col!r}. "
                    "Fit SectorHeadForecaster first and pass sector_preds=oof."
                )

        # Base feature list (without the injected pred_sector_* — added per-h).
        _, base_cols = make_design_matrix(
            ordered,
            feature_cols=self.feature_cols_arg,
            sector_col=self.sector_col,
        )
        self.feature_cols_ = base_cols

        folds = rolling_ts_folds(
            ordered, time_col=self.time_col, n_folds=self.n_folds
        )

        oof_pred: Dict[str, np.ndarray] = {}
        oof_resid: Dict[str, np.ndarray] = {}
        for h in self.horizons:
            oof_pred[f"pred_h{h}"] = np.full(len(ordered), np.nan, dtype=float)
            oof_resid[f"pred_resid_h{h}"] = np.full(len(ordered), np.nan, dtype=float)

        for h in self.horizons:
            y_col = forward_target_col(h)
            ps_col = f"pred_sector_h{h}"
            if y_col not in ordered.columns:
                raise ValueError(
                    f"InstrumentResidualForecaster: missing target {y_col!r}"
                )

            # OOF residual label (never in-sample sector preds).
            resid = ordered[y_col].astype(float) - ordered[ps_col].astype(float)
            ordered[f"resid_h{h}"] = resid

            for train_idx, val_idx in folds:
                y_tr_full = resid.iloc[train_idx]
                tr_mask = y_tr_full.notna().to_numpy()
                if tr_mask.sum() < 2:
                    continue
                tr_pos = train_idx[tr_mask]

                tr_ps = ordered.iloc[tr_pos][ps_col]
                va_ps_all = ordered.iloc[val_idx][ps_col]
                tr_ps_f, _, mean = _fill_with_train_mean(tr_ps, va_ps_all)

                X_tr, _ = make_design_matrix(
                    ordered.iloc[tr_pos],
                    feature_cols=self.feature_cols_,
                    sector_col=self.sector_col,
                )
                X_tr = X_tr.copy()
                X_tr[ps_col] = tr_ps_f.to_numpy()
                y_tr = resid.iloc[tr_pos].astype(float).to_numpy()

                model = _new_xgb(self.params, seed=self.seed + 100 + h)
                model.fit(X_tr, y_tr)

                y_va = resid.iloc[val_idx]
                va_mask = y_va.notna().to_numpy()
                # Predict residual also where residual label is NaN but sector
                # pred exists? No — OOF coverage tracks labeled rows.
                if va_mask.sum() == 0:
                    continue
                va_pos = val_idx[va_mask]
                X_va, _ = make_design_matrix(
                    ordered.iloc[va_pos],
                    feature_cols=self.feature_cols_,
                    sector_col=self.sector_col,
                )
                X_va = X_va.copy()
                _, va_ps_f, _ = _fill_with_train_mean(tr_ps, ordered.iloc[va_pos][ps_col])
                X_va[ps_col] = va_ps_f.to_numpy()
                if self.sector_col in X_tr.columns and self.sector_col in X_va.columns:
                    X_va[self.sector_col] = pd.Categorical(
                        X_va[self.sector_col],
                        categories=X_tr[self.sector_col].cat.categories,
                    )
                pred_resid = model.predict(X_va)
                oof_resid[f"pred_resid_h{h}"][va_pos] = pred_resid
                oof_pred[f"pred_h{h}"][va_pos] = (
                    ordered.iloc[va_pos][ps_col].astype(float).to_numpy() + pred_resid
                )

            # Fill remaining labeled gaps (earliest train-only block).
            labeled = resid.notna().to_numpy()
            missing = labeled & np.isnan(oof_pred[f"pred_h{h}"])
            if missing.any():
                have = labeled & ~np.isnan(oof_pred[f"pred_h{h}"])
                src = np.where(have)[0]
                dst = np.where(missing)[0]
                if len(src) >= 2:
                    tr_ps = ordered.iloc[src][ps_col]
                    tr_ps_f, dst_ps_f, mean = _fill_with_train_mean(
                        tr_ps, ordered.iloc[dst][ps_col]
                    )
                    X_src, _ = make_design_matrix(
                        ordered.iloc[src],
                        feature_cols=self.feature_cols_,
                        sector_col=self.sector_col,
                    )
                    X_src = X_src.copy()
                    X_src[ps_col] = tr_ps_f.to_numpy()
                    y_src = resid.iloc[src].astype(float).to_numpy()
                    filler = _new_xgb(self.params, seed=self.seed + 100 + h + 17)
                    filler.fit(X_src, y_src)
                    X_dst, _ = make_design_matrix(
                        ordered.iloc[dst],
                        feature_cols=self.feature_cols_,
                        sector_col=self.sector_col,
                    )
                    X_dst = X_dst.copy()
                    X_dst[ps_col] = dst_ps_f.to_numpy()
                    if self.sector_col in X_src.columns and self.sector_col in X_dst.columns:
                        X_dst[self.sector_col] = pd.Categorical(
                            X_dst[self.sector_col],
                            categories=X_src[self.sector_col].cat.categories,
                        )
                    pred_resid = filler.predict(X_dst)
                    oof_resid[f"pred_resid_h{h}"][dst] = pred_resid
                    oof_pred[f"pred_h{h}"][dst] = (
                        ordered.iloc[dst][ps_col].astype(float).to_numpy() + pred_resid
                    )

            # Final full-fit on all labeled residual rows.
            all_lab = np.where(labeled)[0]
            if len(all_lab) < 2:
                raise ValueError(
                    f"InstrumentResidualForecaster: horizon h={h} has <2 residual labels"
                )
            tr_ps = ordered.iloc[all_lab][ps_col]
            tr_ps_f, _, mean = _fill_with_train_mean(tr_ps, tr_ps)
            self.pred_sector_means_[h] = mean
            X_all, feat_names = make_design_matrix(
                ordered.iloc[all_lab],
                feature_cols=self.feature_cols_,
                sector_col=self.sector_col,
            )
            X_all = X_all.copy()
            X_all[ps_col] = tr_ps_f.to_numpy()
            y_all = resid.iloc[all_lab].astype(float).to_numpy()
            final = _new_xgb(self.params, seed=self.seed + 100 + h)
            final.fit(X_all, y_all)
            self.models_[h] = final
            # Persist the residual feature scheme: base + pred_sector_hN.
            if h == self.horizons[0]:
                self.feature_cols_ = feat_names  # base only; ps injected at predict

        if return_oof:
            keep = [c for c in (self.time_col, "ticker") if c in ordered.columns]
            oof = ordered[keep].copy()
            for h in self.horizons:
                oof[f"pred_sector_h{h}"] = ordered[f"pred_sector_h{h}"].to_numpy()
                oof[f"pred_resid_h{h}"] = oof_resid[f"pred_resid_h{h}"]
                oof[f"pred_h{h}"] = oof_pred[f"pred_h{h}"]
                oof[f"resid_h{h}"] = ordered[f"resid_h{h}"].to_numpy()
            self.oof_ = oof
        self._fitted = True
        return self

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        sector_preds: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict ``pred_hN`` on new rows given sector-head predictions."""
        if not self._fitted:
            raise RuntimeError("InstrumentResidualForecaster.predict: call fit() first")
        ordered = frame.reset_index(drop=True).copy()
        sp = sector_preds.reset_index(drop=True)
        if len(sp) != len(ordered):
            raise ValueError(
                "InstrumentResidualForecaster.predict: sector_preds length mismatch"
            )
        out = pd.DataFrame(index=ordered.index)
        if self.time_col in ordered.columns:
            out[self.time_col] = ordered[self.time_col].to_numpy()
        if "ticker" in ordered.columns:
            out["ticker"] = ordered["ticker"].to_numpy()

        for h, model in self.models_.items():
            ps_col = f"pred_sector_h{h}"
            if ps_col not in sp.columns:
                raise ValueError(f"predict: sector_preds missing {ps_col!r}")
            mean = self.pred_sector_means_.get(h, 0.0)
            ps = sp[ps_col].astype(float).fillna(mean)
            X, _ = make_design_matrix(
                ordered,
                feature_cols=[c for c in self.feature_cols_ if c != ps_col],
                sector_col=self.sector_col,
            )
            X = X.copy()
            X[ps_col] = ps.to_numpy()
            if self.sector_col in X.columns:
                X[self.sector_col] = X[self.sector_col].astype("category")
            pred_resid = model.predict(X)
            out[f"pred_sector_h{h}"] = ps.to_numpy()
            out[f"pred_resid_h{h}"] = pred_resid
            out[f"pred_h{h}"] = ps.to_numpy() + pred_resid
        return out

    def oof_predictions(self) -> pd.DataFrame:
        if self.oof_ is None:
            raise RuntimeError("InstrumentResidualForecaster.oof_predictions: no OOF")
        return self.oof_.copy()
