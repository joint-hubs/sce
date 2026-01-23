"""
@module: sce.cleanup
@depends: sce.config
@exports: FeatureCleanupPipeline, CleanupReport
@paper_ref: Not in paper (utility module)
@data_flow: features -> cleanup steps -> reduced feature set
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sce.config import CleanupConfig

logger = logging.getLogger(__name__)


@dataclass
class CleanupReport:
    """Report of cleanup operations performed."""

    original_features: int
    final_features: int
    constant_removed: List[str] = field(default_factory=list)
    leakage_removed: List[str] = field(default_factory=list)
    leakage_warned: List[str] = field(default_factory=list)
    correlation_removed: List[str] = field(default_factory=list)
    vif_removed: List[str] = field(default_factory=list)
    hierarchy_removed: List[str] = field(default_factory=list)

    @property
    def total_removed(self) -> int:
        return self.original_features - self.final_features

    @property
    def removal_rate(self) -> float:
        if self.original_features == 0:
            return 0.0
        return self.total_removed / self.original_features


class FeatureCleanupPipeline:
    """
    Comprehensive feature cleanup pipeline.

    Removes:
    1. Constant/near-zero variance features
    2. Features with suspiciously high target correlation
    3. Highly correlated feature pairs
    4. Optional VIF multicollinearity
    5. Hierarchical redundancy for SCE features
    """

    def __init__(self, config: Optional[CleanupConfig] = None) -> None:
        self.config = config or CleanupConfig()
        self.removed_features_: List[str] = []
        self.report_: Optional[CleanupReport] = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_col: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, CleanupReport]:
        """Fit cleanup rules and transform features."""
        cleaned = X.copy()
        original_features = len(cleaned.columns)

        constant_removed = self._remove_constant_features(cleaned)
        cleaned = cleaned.drop(columns=constant_removed, errors="ignore")

        leakage_removed, leakage_warned = self._detect_and_remove_leakage(cleaned, y)
        cleaned = cleaned.drop(columns=leakage_removed, errors="ignore")

        correlation_removed = self._remove_correlated_features(cleaned, y, target_col)
        cleaned = cleaned.drop(columns=correlation_removed, errors="ignore")

        vif_removed = self._remove_high_vif_features(cleaned)
        cleaned = cleaned.drop(columns=vif_removed, errors="ignore")

        hierarchy_removed = self._remove_hierarchical_redundancy(cleaned, target_col)
        cleaned = cleaned.drop(columns=hierarchy_removed, errors="ignore")

        removed_features = (
            constant_removed
            + leakage_removed
            + correlation_removed
            + vif_removed
            + hierarchy_removed
        )
        removed_features = [f for f in removed_features if f in X.columns]

        self.removed_features_ = removed_features
        self.report_ = CleanupReport(
            original_features=original_features,
            final_features=len(cleaned.columns),
            constant_removed=constant_removed,
            leakage_removed=leakage_removed,
            leakage_warned=leakage_warned,
            correlation_removed=correlation_removed,
            vif_removed=vif_removed,
            hierarchy_removed=hierarchy_removed,
        )

        logger.info(
            "Cleanup complete: %s -> %s features (removed %s)",
            original_features,
            len(cleaned.columns),
            self.report_.total_removed,
        )
        return cleaned, self.report_

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply previously learned removals."""
        if not self.removed_features_:
            return X.copy()
        return X.drop(columns=[c for c in self.removed_features_ if c in X.columns])

    def _numeric_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Return numeric-only view with infinities mapped to NaN.

        Note: We intentionally avoid imputing NaN values here to prevent
        altering feature distributions inside the cleanup pipeline. Pandas
        correlation/variance routines handle NaN values pairwise.
        """
        numeric = X.select_dtypes(include=[np.number]).copy()
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        return numeric

    def _remove_constant_features(self, X: pd.DataFrame) -> List[str]:
        numeric = self._numeric_frame(X)
        variances = numeric.var()
        removed = variances[variances < self.config.min_variance].index.tolist()
        if removed:
            logger.info("Removed %s constant/near-zero variance features", len(removed))
        return removed

    def _detect_and_remove_leakage(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[List[str], List[str]]:
        if not self.config.leakage_enabled:
            return [], []

        numeric = self._numeric_frame(X)
        correlations = numeric.corrwith(y, method=self.config.correlation_method).abs()
        correlations = correlations.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        leaked = correlations[correlations > self.config.leakage_remove_threshold]
        warned = correlations[
            (correlations > self.config.leakage_warn_threshold)
            & (correlations <= self.config.leakage_remove_threshold)
        ]

        leaked_features = leaked.index.tolist()
        warned_features = warned.index.tolist()

        if leaked_features:
            logger.warning("Removed %s potential leakage features", len(leaked_features))
        if warned_features:
            logger.warning("Flagged %s potential leakage features", len(warned_features))

        return leaked_features, warned_features

    def _remove_correlated_features(
        self, X: pd.DataFrame, y: pd.Series, target_col: Optional[str]
    ) -> List[str]:
        if not self.config.correlation_enabled:
            return []

        numeric = self._numeric_frame(X)
        if numeric.shape[1] < 2:
            return []

        corr_matrix = numeric.corr(method=self.config.correlation_method).abs()
        target_corr = numeric.corrwith(y, method=self.config.correlation_method).abs().fillna(0.0)
        variances = numeric.var()
        removed: List[str] = []

        for _ in range(self.config.correlation_max_iterations):
            pairs = self._find_correlated_pairs(corr_matrix, self.config.correlation_threshold)
            if not pairs:
                break

            feat1, feat2, _ = pairs[0]
            to_drop = self._select_feature_to_drop(
                feat1,
                feat2,
                target_corr,
                variances,
                target_col,
            )

            if to_drop not in numeric.columns:
                break

            removed.append(to_drop)
            numeric = numeric.drop(columns=[to_drop])
            corr_matrix = numeric.corr(method=self.config.correlation_method).abs()

        if removed:
            logger.info("Removed %s correlated features", len(removed))
        return removed

    def _find_correlated_pairs(
        self, corr_matrix: pd.DataFrame, threshold: float
    ) -> List[Tuple[str, str, float]]:
        pairs = []
        cols = corr_matrix.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                corr = corr_matrix.iloc[i, j]
                if pd.isna(corr):
                    continue
                if corr > threshold:
                    pairs.append((cols[i], cols[j], corr))
        return sorted(pairs, key=lambda x: x[2], reverse=True)

    def _select_feature_to_drop(
        self,
        feat1: str,
        feat2: str,
        target_corr: pd.Series,
        variances: pd.Series,
        target_col: Optional[str],
    ) -> str:
        strategy = self.config.correlation_drop_strategy

        if strategy == "lower_variance":
            return feat1 if variances.get(feat1, 0) < variances.get(feat2, 0) else feat2

        if strategy == "hierarchy" and target_col:
            depth1 = self._sce_feature_depth(feat1, target_col)
            depth2 = self._sce_feature_depth(feat2, target_col)
            if depth1 != depth2:
                return feat1 if depth1 < depth2 else feat2

        if strategy == "first":
            return feat1

        # Default: lower target correlation
        return feat1 if target_corr.get(feat1, 0) < target_corr.get(feat2, 0) else feat2

    def _remove_high_vif_features(self, X: pd.DataFrame) -> List[str]:
        if not self.config.vif_enabled:
            return []

        try:
            import importlib

            module = importlib.import_module("statsmodels.stats.outliers_influence")
            variance_inflation_factor = module.variance_inflation_factor
        except Exception:
            logger.warning("statsmodels not available; skipping VIF cleanup")
            return []

        numeric = self._numeric_frame(X)
        removed: List[str] = []

        for _ in range(self.config.vif_max_iterations):
            if numeric.shape[1] < 2:
                break

            vif_values = []
            for i in range(numeric.shape[1]):
                try:
                    vif_values.append(variance_inflation_factor(numeric.values, i))
                except Exception:
                    vif_values.append(np.inf)

            max_vif = float(np.max(vif_values)) if vif_values else 0.0
            if max_vif <= self.config.vif_threshold:
                break

            to_drop = numeric.columns[int(np.argmax(vif_values))]
            removed.append(to_drop)
            numeric = numeric.drop(columns=[to_drop])

        if removed:
            logger.info("Removed %s high-VIF features", len(removed))
        return removed

    def _remove_hierarchical_redundancy(
        self, X: pd.DataFrame, target_col: Optional[str]
    ) -> List[str]:
        if not self.config.hierarchy_enabled or not target_col:
            return []

        numeric = self._numeric_frame(X)
        removed: List[str] = []
        features = list(numeric.columns)

        for i, feat in enumerate(features):
            for other in features[i + 1 :]:
                pair = self._hierarchical_pair(feat, other, target_col)
                if pair is None:
                    continue

                child, parent = pair
                if child not in numeric.columns or parent not in numeric.columns:
                    continue

                corr = abs(numeric[child].corr(numeric[parent]))
                if np.isnan(corr) or corr < self.config.hierarchy_corr_threshold:
                    continue

                to_drop = parent if self.config.hierarchy_prefer == "child" else child
                if to_drop not in removed:
                    removed.append(to_drop)

        if removed:
            logger.info("Removed %s hierarchy-redundant features", len(removed))
        return removed

    def _hierarchical_pair(
        self, feat1: str, feat2: str, target_col: str
    ) -> Optional[Tuple[str, str]]:
        parsed1 = self._parse_sce_feature(feat1, target_col)
        parsed2 = self._parse_sce_feature(feat2, target_col)
        if parsed1 is None or parsed2 is None:
            return None

        if parsed1["stat"] != parsed2["stat"]:
            return None

        cols1 = parsed1["columns"]
        cols2 = parsed2["columns"]

        if set(cols1).issubset(set(cols2)) and len(cols2) > len(cols1):
            return feat2, feat1  # child, parent
        if set(cols2).issubset(set(cols1)) and len(cols1) > len(cols2):
            return feat1, feat2

        return None

    def _parse_sce_feature(self, feature: str, target_col: str) -> Optional[Dict[str, object]]:
        marker = f"_{target_col}_"
        if marker not in feature:
            return None

        level_name, stat = feature.split(marker, 1)
        columns = level_name.split("__") if level_name else []
        return {"stat": stat, "columns": columns}

    def _sce_feature_depth(self, feature: str, target_col: str) -> int:
        parsed = self._parse_sce_feature(feature, target_col)
        if parsed is None:
            return 0
        return len(parsed["columns"])
