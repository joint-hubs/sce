"""
@module: equity.sce.enrich
@depends: pandas, pathlib, sce, equity.sce.config
@exports: EquityContextEnricher, build_context_config
@paper_ref: docs/plan/2026-07-27_trading_forecaster_prd.md §6.1 (S4)
@data_flow: features (from build_features) + static sectors CSV
            -> prepare (tz, join hierarchy, time_bucket, ret_1d alias)
            -> StatisticalContextEngine.fit_transform (rolling CF)
            -> post-filter interaction levels to equity allow-list
            -> enriched frame (index preserved)

S4.1/S4.2/S4.4 equity SCE wrapper. Does NOT modify vendored ``sce/``.
Interaction cardinality is bounded by post-filtering SCE output columns against
``EquityHierarchyConfig.interactions`` (ADR 0001 interim strategy).

Column naming contract (verified against sce/engine.py:_join_level_stats and
sce/stats.py StatsAggregator)::

    single-col / interaction level:
        ``{level}_{target}_{stat}``
        e.g. ``ticker_ret_1d_mean``, ``sector__time_bucket_ret_1d_mean``
    global level (special-cased join):
        ``global_{target}_{stat}``
        e.g. ``global_ret_1d_mean``
    optional fold-variance suffixes (random CF / include_fold_variance):
        ``{level}_{target}_{stat}_fold_{std|cv|lower|upper}``

``ret_1d`` target is PAST-ONLY: the enricher aliases ``ret_1d = ret_1d_log``
from S3 ``add_returns`` (``log(close[t-1]/close[t-1-n]).shift(1)``). Never
use ``close.pct_change()`` here (forward-looking leak).

``transform_partial`` (S4.4) is PIT-safe: full re-fit of a dedicated
non-cross-fitting engine on ``[train_start, refit_boundary_ts]``, then
``transform`` of new rows strictly after the boundary. See method docstring
for the engine.py line citation.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional, Set, Union

import pandas as pd

from equity.sce.config import (
    DEFAULT_EQUITY_HIERARCHY,
    DEFAULT_INTERACTIONS,
    EquityHierarchyConfig,
)
from sce import ContextConfig, StatisticalContextEngine

logger = logging.getLogger(__name__)

# Known AggregationMethod.value tokens + fold-variance suffixes SCE may attach.
# Used only as a defensive suffix set when stripping a column name back to its
# level; the primary parser uses a target-anchored regex (more precise).
_FOLD_VARIANCE_SUFFIXES: tuple[str, ...] = (
    "fold_std",
    "fold_cv",
    "fold_lower",
    "fold_upper",
)

# Hierarchy columns joined from the static sectors CSV.
_SECTOR_HIERARCHY_COLS: tuple[str, ...] = ("sector", "industry", "mktcap_bucket")

# Default on-disk location of the seed sector map (relative to repo root when
# the package is imported from a source checkout). Callers may pass an
# absolute path or an in-memory DataFrame instead.
_DEFAULT_SECTORS_CSV = (
    Path(__file__).resolve().parents[2] / "configs" / "equity" / "sp500_sectors.csv"
)


def _canonicalize_tz_utc(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Convert ``df[col]`` to UTC (idempotent). Mirrors build.py:52-63."""
    if col not in df.columns:
        return df
    out = df.copy()
    ts = out[col]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        out[col] = pd.to_datetime(ts, utc=True)
        return out
    if ts.dt.tz is None:
        # tz-naive -- assume UTC (the aggregator's / SCE canonical storage tz).
        out[col] = ts.dt.tz_localize("UTC")
    else:
        out[col] = ts.dt.tz_convert("UTC")
    return out


def build_context_config(
    hierarchy: EquityHierarchyConfig = DEFAULT_EQUITY_HIERARCHY,
) -> ContextConfig:
    """Map :class:`EquityHierarchyConfig` → upstream :class:`sce.ContextConfig`.

    Every locked S4 field is set explicitly (no silent defaults for the
    leakage-sensitive knobs). ``include_interactions=True`` so SCE still
    builds pairs; the equity allow-list is enforced by the post-filter.
    """
    # Defensive exclude: any ret_h* forward targets + configured extras.
    exclude: List[str] = list(hierarchy.exclude_cols)

    return ContextConfig(
        target_col=hierarchy.target_col,
        categorical_cols=list(hierarchy.categorical_cols),
        min_group_size=hierarchy.min_group_size,
        use_cross_fitting=True,
        cross_fit_strategy=hierarchy.cross_fit_strategy,  # type: ignore[arg-type]
        time_col=hierarchy.time_col,
        n_folds=hierarchy.n_folds,
        random_state=hierarchy.random_state,
        max_interaction_depth=hierarchy.max_interaction_depth,
        include_relative_features=hierarchy.include_relative_features,
        include_interactions=True,
        include_global_stats=True,
        exclude_cols=exclude,
    )


def _level_from_context_column(col: str, target_col: str) -> Optional[str]:
    """Recover the SCE level name from an enriched column, or ``None``.

    SCE naming (verified)::

        {level}_{target}_{stat}[ _fold_{var} ]

    where ``stat ∈ {mean, median, std, q05, q20, q80, q95, count, ...}`` and
    optional fold-variance suffix is one of ``fold_std|fold_cv|fold_lower|
    fold_upper``. Interaction levels use ``__`` inside ``level``
    (e.g. ``sector__time_bucket_ret_1d_mean``).

    Returns the level string (``"global"``, ``"ticker"``,
    ``"sector__time_bucket"``, ...) when ``col`` matches the naming contract;
    ``None`` when the column is not an SCE context feature (passthrough input
    columns like ``close``, ``ret_1d_log``, hierarchy keys, ...).
    """
    # Anchor on ``_{target_col}_`` so we don't split inside a level that
    # happens to contain the target token (defensive; our levels don't).
    needle = f"_{target_col}_"
    idx = col.find(needle)
    if idx <= 0:
        return None
    level = col[:idx]
    rest = col[idx + len(needle) :]
    if not rest:
        return None
    # rest is "{stat}" or "{stat}_fold_{var}". Accept either.
    # stat token: letters/digits only (AggregationMethod.value).
    if re.fullmatch(r"[A-Za-z0-9]+", rest):
        return level
    for fold_suf in _FOLD_VARIANCE_SUFFIXES:
        # e.g. rest == "mean_fold_std"
        if rest.endswith("_" + fold_suf):
            stat_part = rest[: -(len(fold_suf) + 1)]
            if re.fullmatch(r"[A-Za-z0-9]+", stat_part):
                return level
    return None


class EquityContextEnricher:
    """Wrap :class:`sce.StatisticalContextEngine` for the equity manifold.

    Parameters
    ----------
    hierarchy:
        Frozen knobs (target, categoricals, allow-listed interactions, CF).
    sectors:
        Static hierarchy map. Accepts:

        * ``None`` — load :file:`configs/equity/sp500_sectors.csv` from the
          repo root (source checkout layout).
        * :class:`pathlib.Path` / ``str`` — CSV path with columns
          ``ticker,sector,industry,mktcap_bucket``.
        * :class:`pandas.DataFrame` — already-loaded frame with those columns.
          Unit tests pass a synthetic frame this way.
    """

    def __init__(
        self,
        hierarchy: EquityHierarchyConfig = DEFAULT_EQUITY_HIERARCHY,
        sectors: Union[pd.DataFrame, Path, str, None] = None,
    ) -> None:
        self.hierarchy = hierarchy
        self._sectors_df = self._load_sectors(sectors)
        self._engine: Optional[StatisticalContextEngine] = None
        self._partial_engine: Optional[StatisticalContextEngine] = None
        self._last_fold_timestamps: List[dict[str, Any]] = []
        self._prepared_index: Optional[pd.Index] = None
        # Prepared frame from the last fit_transform; used by transform_partial
        # to slice the fit window without re-running sector join / aliasing.
        self._prepared_features: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # sectors IO
    # ------------------------------------------------------------------
    @staticmethod
    def _load_sectors(
        sectors: Union[pd.DataFrame, Path, str, None],
    ) -> pd.DataFrame:
        if sectors is None:
            path = _DEFAULT_SECTORS_CSV
            if not path.is_file():
                raise FileNotFoundError(
                    f"Default sectors CSV not found at {path}. Pass an explicit "
                    "sectors DataFrame/Path to EquityContextEnricher."
                )
            df = pd.read_csv(path, comment="#")
        elif isinstance(sectors, (str, Path)):
            df = pd.read_csv(sectors, comment="#")
        elif isinstance(sectors, pd.DataFrame):
            df = sectors.copy()
        else:
            raise TypeError(
                "sectors must be None, Path/str, or DataFrame; "
                f"got {type(sectors).__name__}"
            )

        required = {"ticker", *_SECTOR_HIERARCHY_COLS}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"sectors frame missing required columns {sorted(missing)}; "
                f"have {list(df.columns)}"
            )
        # One row per ticker; keep first if duplicates slip in.
        out = (
            df.loc[:, ["ticker", *_SECTOR_HIERARCHY_COLS]]
            .drop_duplicates(subset=["ticker"], keep="first")
            .copy()
        )
        for col in _SECTOR_HIERARCHY_COLS:
            out[col] = out[col].fillna("unknown").astype(str)
            out.loc[out[col].str.strip().eq(""), col] = "unknown"
        out["ticker"] = out["ticker"].astype(str)
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def build_context_config(self) -> ContextConfig:
        """Build the upstream :class:`~sce.ContextConfig` for this enricher."""
        return build_context_config(self.hierarchy)

    def _build_refit_context_config(self) -> ContextConfig:
        """Build a non-cross-fitting :class:`~sce.ContextConfig` for partial re-fit.

        ``transform_partial`` fits on a known train window and transforms future
        rows. Out-of-fold / rolling CF is not applicable (the fold concept is
        for in-sample enrichment of the same frame). SCE ``fit()`` already
        computes an in-sample ``_stats_dict`` regardless of
        ``use_cross_fitting`` (see ``sce/engine.py`` ``fit`` ~L143-160 and
        ``fit_transform`` ~L290-295 where CF only branches inside
        ``fit_transform``), but we still force CF off here so the intent is
        explicit and so a future engine change cannot silently alter partial
        behaviour.
        """
        # ContextConfig is a stdlib dataclass (sce/config.py); replace is enough.
        return replace(
            self.build_context_config(),
            use_cross_fitting=False,
            cross_fit_strategy="off",
        )

    def _allowed_levels(self) -> Set[str]:
        """Set of SCE level names that survive the post-filter.

        Always keeps ``global`` + every single-col categorical. Interaction
        levels are kept iff their ordered tuple is in
        ``hierarchy.interactions`` (joined with ``"__"``).
        """
        levels: Set[str] = {"global"}
        levels.update(self.hierarchy.categorical_cols)
        for combo in self.hierarchy.interactions:
            levels.add("__".join(combo))
        return levels

    # ------------------------------------------------------------------
    # prepare
    # ------------------------------------------------------------------
    def _prepare(self, features: pd.DataFrame) -> pd.DataFrame:
        """Canonicalize tz, join hierarchy, add ``time_bucket`` + target alias.

        Raises
        ------
        ValueError
            If the (aliased) target column is still missing after prepare,
            or if required input columns (``ticker``, ``period_close_ts``)
            are absent.
        """
        if "ticker" not in features.columns:
            raise ValueError("features frame missing required column 'ticker'")
        if self.hierarchy.time_col not in features.columns:
            raise ValueError(
                f"features frame missing required time column "
                f"{self.hierarchy.time_col!r}"
            )

        out = features.copy()
        # Preserve caller index explicitly (SCE fit_transform also preserves
        # it; we stash it so tests can assert against the prepared frame).
        # pandas.DataFrame.merge drops the left index — restore after join so
        # transform_partial / fit_transform keep the caller's index contract
        # even when new_rows is a non-contiguous .loc slice.
        original_index = out.index.copy()
        self._prepared_index = original_index

        # 1. tz → UTC (mirror build.py _canonicalize_tz_utc).
        out = _canonicalize_tz_utc(out, self.hierarchy.time_col)

        # 2. left-join sectors on ticker; missing hierarchy → "unknown".
        # Drop any pre-existing hierarchy cols so the join is clean (tests may
        # pass partial frames; production features from build_features do not
        # carry sector/* today).
        drop_existing = [c for c in _SECTOR_HIERARCHY_COLS if c in out.columns]
        if drop_existing:
            out = out.drop(columns=drop_existing)
        out["ticker"] = out["ticker"].astype(str)
        out = out.merge(self._sectors_df, on="ticker", how="left", sort=False)
        # left/sort=False keeps row order; reattach caller index post-merge.
        out.index = original_index
        for col in _SECTOR_HIERARCHY_COLS:
            out[col] = out[col].fillna("unknown").astype(str)

        # 3. calendar-month time_bucket (string period label, e.g. "2024-01").
        # to_period drops tz; convert to UTC-naive first so the month label is
        # calendar-UTC (matches the canonicalized period_close_ts).
        ts = out[self.hierarchy.time_col]
        ts_naive = ts.dt.tz_convert("UTC").dt.tz_localize(None) if ts.dt.tz is not None else ts
        out["time_bucket"] = ts_naive.dt.to_period("M").astype(str)

        # 4. target alias: ret_1d ← ret_1d_log (past-only; see module docstring).
        # After aliasing, DROP ret_1d_log so it cannot leak into select_dtypes /
        # Ridge design matrices as a silent byte-identical copy of the target.
        target = self.hierarchy.target_col
        if target == "ret_1d":
            if "ret_1d" not in out.columns:
                if "ret_1d_log" not in out.columns:
                    raise ValueError(
                        "target alias requested (ret_1d) but neither 'ret_1d' nor "
                        "'ret_1d_log' is present on the features frame. Run "
                        "equity.features.build_features first."
                    )
                out["ret_1d"] = out["ret_1d_log"]
            if "ret_1d_log" in out.columns:
                out = out.drop(columns=["ret_1d_log"])

        # 5. assert target present.
        if target not in out.columns:
            raise ValueError(
                f"target column {target!r} missing after prepare; available="
                f"{list(out.columns)}"
            )

        # Defensive: drop any exclude_cols / ret_h* forwards from the prepared
        # frame so they cannot leak into SCE auto paths (we pass categorical
        # cols explicitly, but cleaner to keep them out of X entirely).
        # S5 multi-horizon forecaster labels (ret_h1/5/10/21/63) MUST stay out
        # of the SCE feature block — this regex drop is the backstop.
        drop_fwd = [
            c
            for c in out.columns
            if c in self.hierarchy.exclude_cols
            or (c.startswith("ret_h") and c != target)
        ]
        if drop_fwd:
            out = out.drop(columns=drop_fwd)

        return out

    # ------------------------------------------------------------------
    # post-filter
    # ------------------------------------------------------------------
    def _post_filter_interactions(self, enriched: pd.DataFrame) -> pd.DataFrame:
        """Drop SCE interaction columns whose level is not allow-listed.

        Single-col levels + ``global`` always survive. Input/passthrough
        columns (no parseable level) always survive.
        """
        allowed = self._allowed_levels()
        target = self.hierarchy.target_col
        drop: List[str] = []
        for col in enriched.columns:
            level = _level_from_context_column(col, target)
            if level is None:
                continue  # passthrough feature / hierarchy key
            # Interaction levels contain "__"; singles + global do not (our
            # categorical names are underscore-free by contract).
            if "__" in level and level not in allowed:
                drop.append(col)
            elif level not in allowed and level != "global":
                # A single-col level somehow outside categorical_cols (should
                # not happen with explicit config); drop defensively.
                drop.append(col)
        if drop:
            logger.info(
                "equity SCE post-filter dropping %d non-allow-listed columns "
                "(e.g. %s)",
                len(drop),
                drop[:5],
            )
            return enriched.drop(columns=drop)
        return enriched

    # ------------------------------------------------------------------
    # fit_transform
    # ------------------------------------------------------------------
    def fit_transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Prepare ``features``, run rolling-CF SCE, post-filter interactions.

        Returns the enriched frame with the original input index preserved.
        Side-effects: stores ``self._engine``, ``self._prepared_features``
        (for :meth:`transform_partial`), and ``self._last_fold_timestamps``
        (copied from the private SCE attr for later diagnostics; treat as
        read-only).
        """
        prepared = self._prepare(features)
        prepared_index = prepared.index.copy()
        # Remember prepared panel so transform_partial can slice a fit window
        # without re-joining sectors / recomputing time_bucket / ret_1d alias.
        self._prepared_features = prepared

        cfg = self.build_context_config()
        engine = StatisticalContextEngine(cfg)
        out = engine.fit_transform(prepared)

        # Index contract: SCE promises to preserve X.index; re-assert so a
        # future upstream regression fails loudly here rather than silently
        # poisoning a downstream join.
        if not out.index.equals(prepared_index):
            out = out.copy()
            out.index = prepared_index

        out = self._post_filter_interactions(out)

        self._engine = engine
        # Private upstream attr; exposed read-only for S4 diagnostics. Do not
        # rely on it being part of the public SCE API.
        self._last_fold_timestamps = list(
            getattr(engine, "_last_fold_timestamps", []) or []
        )
        return out

    # ------------------------------------------------------------------
    # transform_partial (S4.4)
    # ------------------------------------------------------------------
    def transform_partial(
        self,
        new_rows: pd.DataFrame,
        *,
        refit_boundary_ts: Any,
        train_start: Any = None,
    ) -> pd.DataFrame:
        """Re-fit SCE on ``[train_start, refit_boundary_ts]`` and transform new rows.

        PIT-safe full re-fit (NOT incremental). Context stats joined onto
        ``new_rows`` come ONLY from the fit window; the new rows themselves
        never enter ``_stats_dict``.

        Parameters
        ----------
        new_rows:
            Features frame (same schema as :meth:`fit_transform` input) whose
            ``period_close_ts`` values are strictly **after**
            ``refit_boundary_ts``.
        refit_boundary_ts:
            Inclusive upper bound of the fit window
            (``period_close_ts <= refit_boundary_ts``).
        train_start:
            Optional inclusive lower bound of the fit window. ``None`` (default)
            = expanding window from the earliest prepared row.

        Returns
        -------
        pd.DataFrame
            ``new_rows`` enriched with post-filtered context columns. Index of
            the prepared ``new_rows`` is preserved.

        Raises
        ------
        RuntimeError
            If :meth:`fit_transform` has not been called yet (no prepared panel
            to slice the fit window from).
        ValueError
            If the fit window is empty, or any ``new_rows`` timestamp is not
            strictly after ``refit_boundary_ts``.

        Notes
        -----
        Engine behaviour (verified against ``sce/engine.py``):

        * ``fit(X)`` (L143-160) always computes an in-sample ``_stats_dict`` on
          all of ``X`` and sets ``_fitted=True``. It does **not** branch on
          ``use_cross_fitting``.
        * Cross-fitting only runs inside ``fit_transform`` when
          ``use_cross_fitting=True`` (L290-295 → ``_fit_transform_cross_fitted``).
        * ``transform(X)`` (L164+) joins the already-fitted ``_stats_dict`` onto
          ``X`` without updating stats.

        Therefore ``fit(window).transform(new_rows)`` is the correct PIT-safe
        pattern: window stats are in-sample on the train set; new rows only
        receive a join. We additionally force a non-CF :class:`ContextConfig`
        via :meth:`_build_refit_context_config` so partial re-fit intent stays
        explicit (``cross_fit_strategy='off'``, ``use_cross_fitting=False``).
        """
        if self._prepared_features is None:
            raise RuntimeError(
                "transform_partial requires fit_transform() first so the "
                "prepared feature panel is available to slice the fit window. "
                "Call enricher.fit_transform(features) before transform_partial."
            )

        time_col = self.hierarchy.time_col
        prepared = self._prepared_features
        boundary = pd.Timestamp(refit_boundary_ts)
        # Match prepared tz (canonicalized to UTC in _prepare).
        prepared_tz = prepared[time_col].dt.tz
        if prepared_tz is not None and boundary.tzinfo is None:
            boundary = boundary.tz_localize(prepared_tz)
        elif prepared_tz is not None and boundary.tzinfo is not None:
            boundary = boundary.tz_convert(prepared_tz)
        elif prepared_tz is None and boundary.tzinfo is not None:
            boundary = boundary.tz_convert("UTC").tz_localize(None)

        fit_mask = prepared[time_col] <= boundary
        if train_start is not None:
            start = pd.Timestamp(train_start)
            if prepared_tz is not None and start.tzinfo is None:
                start = start.tz_localize(prepared_tz)
            elif prepared_tz is not None and start.tzinfo is not None:
                start = start.tz_convert(prepared_tz)
            elif prepared_tz is None and start.tzinfo is not None:
                start = start.tz_convert("UTC").tz_localize(None)
            fit_mask = fit_mask & (prepared[time_col] >= start)

        window_df = prepared.loc[fit_mask]
        if window_df.empty:
            raise ValueError(
                f"transform_partial fit window is empty for "
                f"refit_boundary_ts={boundary!r}, train_start={train_start!r}."
            )
        min_gs = self.hierarchy.min_group_size
        if len(window_df) < min_gs:
            logger.warning(
                "transform_partial fit window has %d rows < min_group_size=%d; "
                "many group levels will drop / back off to parent.",
                len(window_df),
                min_gs,
            )

        new_prepared = self._prepare(new_rows)
        new_ts = new_prepared[time_col]
        if not (new_ts > boundary).all():
            bad = int((new_ts <= boundary).sum())
            raise ValueError(
                f"transform_partial requires all new_rows[{time_col!r}] > "
                f"refit_boundary_ts ({boundary!r}); found {bad} row(s) on or "
                "before the boundary (would leak future rows into enrichment "
                "of past / on-boundary timestamps)."
            )

        # Full re-fit on the train window, then transform only the future rows.
        # See method docstring for the engine.py fit/transform citation.
        cfg = self._build_refit_context_config()
        new_engine = StatisticalContextEngine(cfg)
        new_engine.fit(window_df)
        out = new_engine.transform(new_prepared)

        # Prefer the caller's new_rows index (survives _prepare merge restore);
        # fall back to prepared index if lengths match but identity differs.
        caller_index = new_rows.index
        if len(out) == len(caller_index) and not out.index.equals(caller_index):
            out = out.copy()
            out.index = caller_index.copy()
        elif not out.index.equals(new_prepared.index):
            out = out.copy()
            out.index = new_prepared.index.copy()

        out = self._post_filter_interactions(out)
        # Store the refit engine for diagnostics (does not clobber CF engine
        # from fit_transform unless the caller prefers the latest). Keep the
        # CF engine as self._engine from fit_transform; expose refit separately.
        self._partial_engine = new_engine
        return out


# Re-export config symbols so ``from equity.sce.enrich import ...`` works for
# callers that prefer a single import site; the package __init__ also exposes
# them lazily.
__all__ = [
    "EquityContextEnricher",
    "EquityHierarchyConfig",
    "DEFAULT_EQUITY_HIERARCHY",
    "DEFAULT_INTERACTIONS",
    "build_context_config",
    "_level_from_context_column",
]
