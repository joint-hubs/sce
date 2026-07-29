"""
@module: equity.metrics.sharpe
@depends: numpy, pandas, equity.metrics.accuracy
@exports: decile_long_short_returns, sharpe_ratio, sortino_ratio,
          select_horizon, aggregate_portfolio
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §8 (S6.3 portfolio)
@data_flow: per-horizon VAL prediction frames -> decile L/S series
            -> Sharpe/Sortino -> chosen horizon

S6.3 portfolio metrics. Annualization assumes trading-day rows (periods_per_year=252):
Sharpe = mean(r)/std(r) * sqrt(252); Sortino uses downside deviation of r<0 only.

Decile long/short is **per-timestamp cross-section**: rank by pred, long top bin /
short bottom bin, equal-weight within bin. When a cross-section has n < 10 instruments,
we use ``n_bins = min(10, n)`` equal-rank bins (never crash). Cross-sections with
n < 2 or fewer than 2 usable bins after tie collapse are skipped (NaN).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from equity.metrics.accuracy import directional_hit_rate

ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]


def _finite_1d(returns: ArrayLike) -> np.ndarray:
    r = np.asarray(returns, dtype=float).reshape(-1)
    return r[np.isfinite(r)]


def decile_long_short_returns(
    predictions: pd.DataFrame,
    *,
    ret_col: str,
    pred_col: str,
    time_col: str,
) -> pd.Series:
    """Per-timestamp equal-weight top-minus-bottom decile portfolio returns.

    For each distinct ``time_col`` value, instruments with finite ``pred_col`` and
    ``ret_col`` are ranked by prediction into up to 10 equal-count bins
    (``n_bins = min(10, n)`` when n < 10). Portfolio return =
    mean(ret of top bin) − mean(ret of bottom bin).

    Returns
    -------
    pd.Series
        Indexed by timestamp (same dtype as ``time_col``), NaN where the
        cross-section could not form a long/short (n < 2 or collapsed bins).
    """
    if predictions is None or len(predictions) == 0:
        return pd.Series(dtype=float)

    need = {ret_col, pred_col, time_col}
    missing = need - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing columns: {sorted(missing)}")

    frame = predictions[[time_col, ret_col, pred_col]].copy()
    frame[ret_col] = pd.to_numeric(frame[ret_col], errors="coerce")
    frame[pred_col] = pd.to_numeric(frame[pred_col], errors="coerce")

    def _one_ts(g: pd.DataFrame) -> float:
        valid = g.dropna(subset=[ret_col, pred_col])
        n = int(len(valid))
        if n < 2:
            return float("nan")
        n_bins = min(10, n)
        # Rank first so qcut gets unique scores even on ties; method=first breaks ties
        # deterministically by row order within the group.
        ranks = valid[pred_col].rank(method="first", ascending=True)
        try:
            bins = pd.qcut(ranks, n_bins, labels=False, duplicates="drop")
        except ValueError:
            return float("nan")
        bins = pd.Series(bins, index=valid.index)
        if bins.nunique(dropna=True) < 2:
            return float("nan")
        top = int(bins.max())
        bot = int(bins.min())
        long_ret = float(valid.loc[bins == top, ret_col].mean())
        short_ret = float(valid.loc[bins == bot, ret_col].mean())
        if not (np.isfinite(long_ret) and np.isfinite(short_ret)):
            return float("nan")
        return long_ret - short_ret

    # Iterate groups explicitly (avoids FutureWarning on include_groups and
    # keeps the return type a plain float Series indexed by timestamp).
    parts: dict = {}
    for key, g in frame.groupby(time_col, sort=True):
        parts[key] = _one_ts(g)
    out = pd.Series(parts, dtype=float, name="ls_ret")
    return out


def _is_vanishing_std(std: float, scale: float = 1.0) -> bool:
    """True when std is non-finite, zero, or numerically vanishing vs ``scale``."""
    if not np.isfinite(std) or std <= 0.0:
        return True
    # Guard float-noise std on near-constant series (e.g. [0.1,0.1,0.1]).
    ref = max(1.0, abs(float(scale)))
    return bool(std < 1e-12 * ref)


def sharpe_ratio(returns: ArrayLike, *, periods_per_year: int = 252) -> float:
    """Annualized Sharpe: mean/std * sqrt(periods_per_year). NaN if std==0 or empty."""
    r = _finite_1d(returns)
    if r.size == 0:
        return float("nan")
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if r.size >= 2 else 0.0
    if _is_vanishing_std(std, scale=mean):
        return float("nan")
    return float(mean / std * np.sqrt(float(periods_per_year)))


def sortino_ratio(returns: ArrayLike, *, periods_per_year: int = 252) -> float:
    """Annualized Sortino using downside deviation of returns < 0 only.

    NaN when there are no finite returns, no strict-negative returns, or
    downside std is zero.
    """
    r = _finite_1d(returns)
    if r.size == 0:
        return float("nan")
    downside = r[r < 0.0]
    if downside.size == 0:
        return float("nan")
    if downside.size == 1:
        dstd = float(abs(downside[0]))
    else:
        dstd = float(np.std(downside, ddof=1))
    mean = float(np.mean(r))
    if _is_vanishing_std(dstd, scale=mean if mean != 0 else 1.0):
        return float("nan")
    return float(mean / dstd * np.sqrt(float(periods_per_year)))


def _concat_horizon_frames(
    per_fold_predictions: Sequence[Mapping[int, pd.DataFrame]],
    horizon: int,
    *,
    split: Optional[str] = "val",
) -> pd.DataFrame:
    """Stack per-fold frames for one horizon; optionally filter ``split`` column."""
    chunks: List[pd.DataFrame] = []
    h = int(horizon)
    for fold_map in per_fold_predictions:
        if h not in fold_map:
            continue
        frame = fold_map[h]
        if frame is None or len(frame) == 0:
            continue
        if split is not None and "split" in frame.columns:
            frame = frame.loc[frame["split"].astype(str) == str(split)]
        if len(frame) == 0:
            continue
        chunks.append(frame)
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _criterion_on_frame(
    frame: pd.DataFrame,
    h: int,
    *,
    criterion: str,
    periods_per_year: int,
    time_col: str,
) -> float:
    """Score one concatenated prediction frame under ``criterion``."""
    ret_col = f"ret_h{h}"
    pred_col = f"pred_h{h}"
    crit = str(criterion).lower()
    if crit == "hit_rate":
        if frame.empty or ret_col not in frame.columns or pred_col not in frame.columns:
            return float("nan")
        return float(directional_hit_rate(frame[ret_col], frame[pred_col]))
    if crit not in {"sharpe", "sortino"}:
        raise ValueError(
            f"criterion must be one of {{'sharpe','sortino','hit_rate'}}; got {criterion!r}"
        )
    if frame.empty or ret_col not in frame.columns or pred_col not in frame.columns:
        return float("nan")
    if time_col not in frame.columns:
        raise ValueError(f"predictions missing time_col {time_col!r}")
    ls = decile_long_short_returns(
        frame, ret_col=ret_col, pred_col=pred_col, time_col=time_col
    )
    if crit == "sharpe":
        return sharpe_ratio(ls, periods_per_year=periods_per_year)
    return sortino_ratio(ls, periods_per_year=periods_per_year)


def select_horizon(
    val_predictions: Union[
        Sequence[Mapping[int, pd.DataFrame]],
        Mapping[int, pd.DataFrame],
        pd.DataFrame,
    ],
    horizons: Sequence[int],
    *,
    criterion: str = "sharpe",
    periods_per_year: int = 252,
    time_col: str = "period_close_ts",
) -> int:
    """Pick the horizon maximizing ``criterion`` on the VAL slice.

    ``val_predictions`` may be:
      * a single wide DataFrame carrying ``pred_h{N}`` / ``ret_h{N}`` columns, or
      * a ``{h: frame}`` mapping, or
      * a sequence of per-fold ``{h: frame}`` mappings (concatenated).

    Ties break toward the **smallest** horizon. Horizons with non-finite scores
    are ignored; if every score is non-finite, returns the smallest horizon.
    """
    hs = [int(h) for h in horizons]
    if not hs:
        raise ValueError("horizons must be non-empty")

    # Normalize to {h: frame}
    frames: Dict[int, pd.DataFrame]
    if isinstance(val_predictions, pd.DataFrame):
        frames = {h: val_predictions for h in hs}
    elif isinstance(val_predictions, Mapping) and (
        not val_predictions
        or all(isinstance(k, (int, np.integer)) for k in val_predictions.keys())
    ):
        # Single fold map {h: df}
        frames = {int(h): val_predictions[h] for h in hs if h in val_predictions}
        for h in hs:
            frames.setdefault(h, pd.DataFrame())
    else:
        # Sequence of fold maps
        frames = {
            h: _concat_horizon_frames(val_predictions, h, split="val")  # type: ignore[arg-type]
            for h in hs
        }

    scores: Dict[int, float] = {}
    for h in hs:
        scores[h] = _criterion_on_frame(
            frames.get(h, pd.DataFrame()),
            h,
            criterion=criterion,
            periods_per_year=periods_per_year,
            time_col=time_col,
        )

    finite = [(h, s) for h, s in scores.items() if np.isfinite(s)]
    if not finite:
        return int(min(hs))
    # max score; tie -> smallest h  (sort by (-score, h))
    finite.sort(key=lambda hs_: (-hs_[1], hs_[0]))
    return int(finite[0][0])


def aggregate_portfolio(
    per_fold_val_predictions: Sequence[Mapping[int, pd.DataFrame]],
    horizons: Sequence[int],
    *,
    criterion: str = "sharpe",
    periods_per_year: int = 252,
    time_col: str = "period_close_ts",
) -> Dict[str, Any]:
    """Per-horizon Sharpe/Sortino on concatenated VAL frames + chosen horizon.

    Returns
    -------
    dict
        ``{chosen_horizon, criterion, periods_per_year,
        per_horizon: {h: {sharpe, sortino, hit_rate}}}``.
    """
    hs = [int(h) for h in horizons]
    per_h: Dict[int, Dict[str, float]] = {}
    for h in hs:
        frame = _concat_horizon_frames(per_fold_val_predictions, h, split="val")
        ret_col = f"ret_h{h}"
        pred_col = f"pred_h{h}"
        if (
            frame.empty
            or ret_col not in frame.columns
            or pred_col not in frame.columns
            or time_col not in frame.columns
        ):
            per_h[h] = {
                "sharpe": float("nan"),
                "sortino": float("nan"),
                "hit_rate": float("nan"),
            }
            continue
        ls = decile_long_short_returns(
            frame, ret_col=ret_col, pred_col=pred_col, time_col=time_col
        )
        per_h[h] = {
            "sharpe": sharpe_ratio(ls, periods_per_year=periods_per_year),
            "sortino": sortino_ratio(ls, periods_per_year=periods_per_year),
            "hit_rate": float(directional_hit_rate(frame[ret_col], frame[pred_col])),
        }

    chosen = select_horizon(
        per_fold_val_predictions,
        hs,
        criterion=criterion,
        periods_per_year=periods_per_year,
        time_col=time_col,
    )
    return {
        "chosen_horizon": int(chosen),
        "criterion": str(criterion),
        "periods_per_year": int(periods_per_year),
        "per_horizon": per_h,
    }
