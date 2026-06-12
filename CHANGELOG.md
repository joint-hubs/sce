# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-06-12

### Changed

- **License**: code relicensed from CC-BY-NC-4.0 to Apache-2.0 (the research
  paper remains CC BY-NC 4.0)
- **Evaluation protocol overhaul** — all previously published benchmark
  numbers were produced under a flawed protocol and are superseded:
  - Train/test split now happens before any enrichment; encoders, pruning and
    cleanup are fit on train only
  - `FeatureCombinationSearch` selects candidates on an internal validation
    split (tail split for temporal data); the test set is touched exactly once
  - Temporal datasets use rolling (monotonic) cross-fit folds; random
    cross-fitting with a temporal split is rejected
- `lightgbm` and `catboost` moved from core dependencies to the new `models`
  extra (`pip install stat-context[models]`); xgboost remains core
- `scikit-learn` floor raised to 1.4
- README rewritten around a copy-paste runnable quickstart and the new
  report-grade benchmark table

### Added

- Leakage diagnostics suite (`scripts/diagnostics/`): permuted-target,
  shuffled-groups, cross-fit A/B, feature dominance
- Run metadata (git SHA, config hash, seed, run grade) saved with every
  experiment; report-grade promotion gate requires a clean git tree and all
  diagnostics passing
- Rolling cross-fit `test_size` clamp so configs tuned for full datasets do
  not crash on smaller samples

### Removed

- `rental_poland_long`, `sales_uae_transactions` and `rental_uae_contracts`
  moved to `configs/experimental/` — they currently fail the leakage
  diagnostics gate and are excluded from the benchmark set

## [0.3.5] - 2026-01-23

### Fixed

- Fixed relative links in README for PyPI compatibility (CONTRIBUTING.md, docs links)
- Added GitHub Pages deployment workflow
- Documentation now deployed to https://joint-hubs.github.io/sce/

### Added

- MkDocs site with Material theme
- API reference documentation (auto-generated from docstrings)
- Getting started guides (installation, quickstart)

## [0.3.4] - 2026-01-23

### Fixed

- Aligned documentation with paper and code
- Updated README default aggregations to match config.py (8 methods)
- Fixed `include_interactions` default from `False` to `True` in docs
- Updated dataset sample counts to match paper Table 1
- Updated experiments.md result tables with current RMSE/R² values
- Fixed quickstart example (replaced non-existent `run_experiment` function)
- Fixed feature naming pattern documentation (`{col}_{target}_{stat}`)
- Corrected cleanup.py paper_ref (feature cleanup not in paper)

## [0.3.3] - 2026-01-22

### Fixed

- Updated citation with correct paper title and authors (Mateusz Stachowicz, Stanisław Halkiewicz)
- Updated abstract to match paper

## [0.3.2] - 2026-01-22

### Fixed

- Corrected dataset sample counts and feature counts in README
- Dataset table now shows: Hier. Cols, Base Feats, +SCE Feats

## [0.3.1] - 2026-01-22

### Changed

- Package renamed to `stat-context` for PyPI publication
- Install via: `pip install stat-context`

## [0.3.0] - 2026-01-19

### Added

- Initial public release of Statistical Context Engineering (SCE)
- Core `StatisticalContextEngine` transformer with scikit-learn compatibility
- **Auto-detection** of categorical columns from DataFrames
- Cross-fitting for leakage-safe context computation
- Hierarchical statistical aggregations (mean, median, std, quantiles, count)
- Global statistics as fallback context
- Hierarchical backoff for small groups
- Four benchmark datasets (Poland rentals, UAE contracts/transactions)
- Comprehensive test suite (40+ tests)
- CI/CD with GitHub Actions
- PyPI release automation
- Documentation and examples

### Configuration Options

- `target_col`: Target column for aggregation
- `categorical_cols`: Manual column specification (optional, auto-detected if not provided)
- `aggregations`: List of aggregation methods
- `use_cross_fitting`: Enable/disable leakage prevention
- `n_folds`: Number of cross-fitting folds
- `min_group_size`: Minimum samples per group
- `include_global_stats`: Add dataset-wide statistics
- `include_interactions`: Add cross-column hierarchies

### Experimental

- `search.py`: Model and feature combination search
- `selection.py`: LM-based feature selection

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.3.5 | 2026-01-23 | GitHub Pages docs, PyPI link fixes |
| 0.3.4 | 2026-01-23 | Documentation alignment with paper |
| 0.3.3 | 2026-01-22 | Citation and author info |
| 0.3.0 | 2026-01-19 | Initial public release with auto-detection |
