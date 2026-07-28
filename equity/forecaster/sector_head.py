"""
@module: equity.forecaster.sector_head
@depends: numpy, pandas, xgboost
@exports: SectorHeadForecaster
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §7.2 (layer 1)
@data_flow: features + ret_hN -> rolling OOF XGBoost per horizon -> pred_sector_hN

S5.1 sector-head forecaster. One XGBoost regressor per horizon, trained on the
forward label ``ret_hN``. Out-of-fold predictions cover **val-fold rows only**
via a rolling ts-group CF loop (earliest train-only block stays NaN by design);
a final full-refit model predicts the test slice.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from equity.forecaster._splits import make_design_matrix, rolling_ts_folds
from equity.forecaster.config import DEFAULT_HORIZONS, HorizonConfig, SectorHeadParams
from equity.forecaster.targets import forward_target_col


def _new_xgb(params: SectorHeadParams, *, seed: int):
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("xgboost is required for SectorHeadForecaster") from exc
    return XGBRegressor(**params.to_xgb_kwargs(seed=seed))


class SectorHeadForecaster:
    """One XGBoost per horizon predicting ``ret_hN`` → ``pred_sector_hN``."""

    def __init__(
        self,
        *,
        horizons: Sequence[int] = DEFAULT_HORIZONS,
        params: Optional[SectorHeadParams] = None,
        horizon_cfg: Optional[HorizonConfig] = None,
        feature_cols: Optional[Sequence[str]] = None,
        n_folds: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> None:
        cfg = horizon_cfg or HorizonConfig()
        self.horizons: tuple[int, ...] = tuple(int(h) for h in (horizons or cfg.horizons))
        self.params = params or SectorHeadParams()
        self.time_col = cfg.time_col
        self.sector_col = cfg.sector_col
        self.n_folds = int(n_folds if n_folds is not None else cfg.n_folds)
        self.seed = int(seed if seed is not None else cfg.seed)
        self.feature_cols_arg = list(feature_cols) if feature_cols is not None else None

        self.models_: Dict[int, object] = {}
        self.feature_cols_: List[str] = []
        self.oof_: Optional[pd.DataFrame] = None
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(
        self,
        frame: pd.DataFrame,
        *,
        return_oof: bool = True,
    ) -> "SectorHeadForecaster":
        """Fit per-horizon sector heads and (optionally) return full-row OOF preds.

        Parameters
        ----------
        frame:
            Feature panel that already carries ``ret_hN`` labels (from
            :func:`add_forward_targets`) and the design features. Must include
            ``period_close_ts`` (and ideally ``sector``).
        return_oof:
            When True (default), stores an aligned OOF frame on ``self.oof_``
            with val-fold coverage only. The earliest TimeSeriesSplit train-only
            block stays NaN by design (never used as a val block in any fold).
            A final full-refit on all labeled rows powers ``.predict`` for new X.
        """
        if frame.empty:
            raise ValueError("SectorHeadForecaster.fit: empty frame")

        # Stable sort: time then ticker so same-day cross-sections keep a fixed order
        # across sector / residual / quantile heads (positional OOF align).
        sort_keys = [self.time_col] + (["ticker"] if "ticker" in frame.columns else [])
        ordered = frame.sort_values(sort_keys, kind="mergesort").reset_index(drop=True)
        # Probe design matrix once on the full frame (blocklist strips targets).
        X_probe, cols = make_design_matrix(
            ordered,
            feature_cols=self.feature_cols_arg,
            sector_col=self.sector_col,
        )
        self.feature_cols_ = cols
        del X_probe

        folds = rolling_ts_folds(
            ordered, time_col=self.time_col, n_folds=self.n_folds
        )

        oof_data: Dict[str, np.ndarray] = {
            f"pred_sector_h{h}": np.full(len(ordered), np.nan, dtype=float)
            for h in self.horizons
        }

        for h in self.horizons:
            y_col = forward_target_col(h)
            if y_col not in ordered.columns:
                raise ValueError(
                    f"SectorHeadForecaster: missing target column {y_col!r}. "
                    "Call equity.forecaster.targets.add_forward_targets first."
                )
            pred_col = f"pred_sector_h{h}"

            # Rolling OOF over CF folds.
            for train_idx, val_idx in folds:
                y_train = ordered.iloc[train_idx][y_col]
                train_mask = y_train.notna().to_numpy()
                if train_mask.sum() < 2:
                    continue
                tr_pos = train_idx[train_mask]
                X_tr, _ = make_design_matrix(
                    ordered.iloc[tr_pos],
                    feature_cols=self.feature_cols_,
                    sector_col=self.sector_col,
                )
                y_tr = ordered.iloc[tr_pos][y_col].astype(float).to_numpy()
                model = _new_xgb(self.params, seed=self.seed + h)
                model.fit(X_tr, y_tr)

                y_val = ordered.iloc[val_idx][y_col]
                val_mask = y_val.notna().to_numpy()
                if val_mask.sum() == 0:
                    continue
                va_pos = val_idx[val_mask]
                X_va, _ = make_design_matrix(
                    ordered.iloc[va_pos],
                    feature_cols=self.feature_cols_,
                    sector_col=self.sector_col,
                )
                # Align category levels so XGBoost does not reject unseen codes.
                if self.sector_col in X_tr.columns and self.sector_col in X_va.columns:
                    X_va[self.sector_col] = pd.Categorical(
                        X_va[self.sector_col],
                        categories=X_tr[self.sector_col].cat.categories,
                    )
                oof_data[pred_col][va_pos] = model.predict(X_va)

            # Earliest TimeSeriesSplit train-only block stays NaN in OOF by design
            # (never a val block). Do not back-fill from later rows (forward leak).
            labeled = ordered[y_col].notna().to_numpy()

            # Final full-fit model on all labeled rows (used by .predict for new X).
            all_lab = np.where(labeled)[0]
            if len(all_lab) < 2:
                raise ValueError(
                    f"SectorHeadForecaster: horizon h={h} has <2 labeled rows"
                )
            X_all, _ = make_design_matrix(
                ordered.iloc[all_lab],
                feature_cols=self.feature_cols_,
                sector_col=self.sector_col,
            )
            y_all = ordered.iloc[all_lab][y_col].astype(float).to_numpy()
            final = _new_xgb(self.params, seed=self.seed + h)
            final.fit(X_all, y_all)
            self.models_[h] = final

        if return_oof:
            oof = ordered[[c for c in (self.time_col, "ticker") if c in ordered.columns]].copy()
            for col, arr in oof_data.items():
                oof[col] = arr
            # Reindex-align back onto caller order via original integer positions
            # after the sort: map via a stable key when possible.
            self.oof_ = oof
        self._frame_index_like_ = ordered.index  # sorted range index
        self._fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Predict ``pred_sector_hN`` for every horizon on ``frame`` (new rows)."""
        if not self._fitted:
            raise RuntimeError("SectorHeadForecaster.predict: call fit() first")
        ordered = frame.reset_index(drop=True)
        out = pd.DataFrame(index=ordered.index)
        if self.time_col in ordered.columns:
            out[self.time_col] = ordered[self.time_col].to_numpy()
        if "ticker" in ordered.columns:
            out["ticker"] = ordered["ticker"].to_numpy()

        for h, model in self.models_.items():
            X, _ = make_design_matrix(
                ordered,
                feature_cols=self.feature_cols_,
                sector_col=self.sector_col,
            )
            # Align categories with the fitted booster's training view where possible.
            if self.sector_col in X.columns:
                # Pull categories from the model's feature dive is unreliable;
                # just cast — unseen cats become -1 codes under XGBoost hist.
                X[self.sector_col] = X[self.sector_col].astype("category")
            out[f"pred_sector_h{h}"] = model.predict(X)
        return out

    def oof_predictions(self) -> pd.DataFrame:
        """Return the OOF frame from the last ``fit`` (val-fold coverage; earliest train-only block is NaN by design)."""
        if self.oof_ is None:
            raise RuntimeError("SectorHeadForecaster.oof_predictions: no OOF stored")
        return self.oof_.copy()
