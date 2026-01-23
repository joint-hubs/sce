# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
