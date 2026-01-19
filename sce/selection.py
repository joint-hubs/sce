"""
@module: sce.selection
@depends: scipy, sklearn
@exports: LMFeatureSelector, compute_lm_statistics, select_significant_features
@paper_ref: Section 4.2 Feature Selection
@data_flow: features -> LM statistics -> p-value filtering -> selected_features
@status: EXPERIMENTAL - Test coverage 17%. Not recommended for production use.

Feature selection using Linear Model statistics.
Two-stage approach:
1. LM p-value filtering for statistical significance
2. Optional XGBoost importance ranking for predictive power

⚠️ WARNING: This module has minimal test coverage (17%) and is considered
experimental. Use at your own risk. Core SCE functionality in engine.py
and stats.py is fully tested and production-ready.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class LMStatistics:
    """Results from linear model feature analysis."""

    feature_stats: pd.DataFrame
    r2: float
    intercept: float
    n_samples: int
    n_features: int


def compute_lm_statistics(
    X: pd.DataFrame, y: pd.Series, features: Optional[List[str]] = None
) -> LMStatistics:
    """
    Compute Linear Model statistics for features.

    Fits OLS regression and computes:
    - Standardized coefficients (β)
    - Standard errors SE(β)
    - T-statistics
    - P-values
    - Correlations with target

    Args:
        X: Feature matrix
        y: Target variable
        features: Subset of features to analyze (default: all)

    Returns:
        LMStatistics with per-feature statistics
    """
    if features is None:
        features = list(X.columns)

    X_subset = X[features].copy()

    # Handle missing values
    X_subset = X_subset.fillna(X_subset.median())
    X_subset = X_subset.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Remove constant columns (zero variance)
    var = X_subset.var()
    non_constant = var[var > 1e-10].index.tolist()
    # Continue with non-constant features only
    X_subset = X_subset[non_constant]
    features_used = non_constant

    if len(features_used) == 0:
        return LMStatistics(
            feature_stats=pd.DataFrame(), r2=0.0, intercept=0.0, n_samples=len(y), n_features=0
        )

    # Standardize for comparable coefficients
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_subset), columns=features_used, index=X_subset.index
    )

    # Fit OLS
    model = LinearRegression()
    model.fit(X_scaled, y)

    n = len(y)
    k = len(features_used)

    # Predictions and residuals
    y_pred = model.predict(X_scaled)
    residuals = y - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    dof = max(n - k - 1, 1)
    mse = ss_res / dof

    # Variance-covariance matrix using pseudo-inverse for robustness
    X_with_const = np.column_stack([np.ones(n), X_scaled.values])
    try:
        XtX_inv = np.linalg.pinv(X_with_const.T @ X_with_const)
        var_coef = mse * XtX_inv.diagonal()
        se = np.sqrt(np.maximum(var_coef[1:], 1e-10))
    except Exception:
        se = np.ones(k) * np.nan

    # T-statistics and p-values
    t_stats = np.where(se > 1e-10, model.coef_ / se, 0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))

    # Correlations with target
    correlations = []
    corr_p_values = []
    for feat in features_used:
        try:
            col = X_subset[feat]
            if col.std() > 1e-10:
                r, p = stats.pearsonr(col, y)
                correlations.append(r)
                corr_p_values.append(p)
            else:
                correlations.append(0.0)
                corr_p_values.append(1.0)
        except Exception:
            correlations.append(np.nan)
            corr_p_values.append(np.nan)

    results = pd.DataFrame(
        {
            "feature": features_used,
            "coefficient": model.coef_,
            "coefficient_abs": np.abs(model.coef_),
            "std_error": se,
            "t_statistic": t_stats,
            "p_value": p_values,
            "significant_005": p_values < 0.05,
            "significant_001": p_values < 0.01,
            "correlation": correlations,
            "correlation_abs": np.abs(correlations),
            "correlation_p_value": corr_p_values,
        }
    )

    # Add constant features back with NaN statistics
    for feat in features:
        if feat not in features_used:
            results = pd.concat(
                [
                    results,
                    pd.DataFrame(
                        [
                            {
                                "feature": feat,
                                "coefficient": 0.0,
                                "coefficient_abs": 0.0,
                                "std_error": np.nan,
                                "t_statistic": 0.0,
                                "p_value": 1.0,
                                "significant_005": False,
                                "significant_001": False,
                                "correlation": 0.0,
                                "correlation_abs": 0.0,
                                "correlation_p_value": 1.0,
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )

    return LMStatistics(
        feature_stats=results.sort_values("p_value"),
        r2=r2,
        intercept=model.intercept_,
        n_samples=n,
        n_features=k,
    )


def select_significant_features(
    X: pd.DataFrame,
    y: pd.Series,
    features: List[str],
    p_threshold: float = 0.05,
    method: str = "backward",
) -> Tuple[List[str], pd.DataFrame]:
    """
    Select statistically significant features.

    Args:
        X: Feature matrix
        y: Target variable
        features: Features to consider
        p_threshold: P-value threshold for significance
        method: "backward" (stepwise elimination) or "filter" (simple threshold)

    Returns:
        Tuple of (selected_features, elimination_history)
    """
    if method == "filter":
        # Simple filtering: keep all with p < threshold
        lm_stats = compute_lm_statistics(X, y, features)
        significant = lm_stats.feature_stats[lm_stats.feature_stats["p_value"] < p_threshold][
            "feature"
        ].tolist()
        return significant, lm_stats.feature_stats

    # Backward stepwise elimination
    selected = [f for f in features if f in X.columns]
    history = []

    while len(selected) > 1:
        X_subset = X[selected].copy().fillna(0)
        X_subset = X_subset.replace([np.inf, -np.inf], 0)

        # Remove constant columns
        var = X_subset.var()
        selected = [f for f in selected if var.get(f, 0) > 1e-10]
        if len(selected) <= 1:
            break
        X_subset = X_subset[selected]

        # Fit linear regression
        model = LinearRegression()
        model.fit(X_subset, y)

        # Calculate p-values
        n = len(y)
        k = len(selected)
        y_pred = model.predict(X_subset)
        residuals = y - y_pred
        mse = np.sum(residuals**2) / max(n - k - 1, 1)

        # Standard errors
        X_with_const = np.column_stack([np.ones(n), X_subset.values])
        try:
            var_coef = mse * np.linalg.pinv(X_with_const.T @ X_with_const).diagonal()
            se = np.sqrt(np.maximum(var_coef[1:], 1e-10))
            t_stats = model.coef_ / se
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
        except Exception:
            break

        # Find feature with highest p-value
        max_p_idx = np.argmax(p_values)
        max_p = p_values[max_p_idx]

        if max_p > p_threshold:
            removed = selected.pop(max_p_idx)
            history.append(
                {
                    "step": len(history) + 1,
                    "removed": removed,
                    "p_value": max_p,
                    "remaining": len(selected),
                }
            )
        else:
            break

    history_df = pd.DataFrame(history) if history else pd.DataFrame()
    return selected, history_df


class LMFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible feature selector based on LM p-values.

    Example:
        >>> selector = LMFeatureSelector(p_threshold=0.05)
        >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(self, p_threshold: float = 0.05, method: str = "backward"):
        self.p_threshold = p_threshold
        self.method = method
        self.selected_features_: Optional[List[str]] = None
        self.lm_stats_: Optional[LMStatistics] = None
        self.elimination_history_: Optional[pd.DataFrame] = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LMFeatureSelector":
        """Fit selector to find significant features."""
        features = list(X.columns)

        # Compute initial LM statistics
        self.lm_stats_ = compute_lm_statistics(X, y, features)

        # Select significant features
        self.selected_features_, self.elimination_history_ = select_significant_features(
            X, y, features, self.p_threshold, self.method
        )

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only selected features."""
        if self.selected_features_ is None:
            raise RuntimeError("Must call fit() before transform()")

        # Only keep features that exist in X
        available = [f for f in self.selected_features_ if f in X.columns]
        return X[available].copy()

    def get_support(self) -> List[str]:
        """Return list of selected feature names."""
        return self.selected_features_ or []
