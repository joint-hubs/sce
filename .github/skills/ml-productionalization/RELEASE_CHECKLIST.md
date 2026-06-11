# Release Checklist

> *"Ship it when you'd bet your reputation on it working."*

---

## When to Release

You're ready for release when:
- ✅ Tests pass on clean install
- ✅ Documentation matches the code
- ✅ Example scripts run without modification
- ✅ Someone else can set it up without asking you questions

---

## The Release Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RELEASE WORKFLOW                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. FREEZE       Lock features, focus on stability                 │
│        │                                                            │
│        ▼                                                            │
│   2. VALIDATE     Run all checks (tests, linting, security)        │
│        │                                                            │
│        ▼                                                            │
│   3. DOCUMENT     Update CHANGELOG, README, version numbers         │
│        │                                                            │
│        ▼                                                            │
│   4. TEST INSTALL Clean environment install test                    │
│        │                                                            │
│        ▼                                                            │
│   5. TAG          Git tag with version number                       │
│        │                                                            │
│        ▼                                                            │
│   6. PUBLISH      Push to PyPI / create GitHub release              │
│        │                                                            │
│        ▼                                                            │
│   7. VERIFY       Confirm installation works from published source  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: FREEZE

### Code Freeze Checklist

- [ ] No new features until after release
- [ ] All PRs for this release are merged
- [ ] `main` branch is stable
- [ ] Dependency versions are pinned

### Create a Release Branch (Optional)

```bash
# For major releases, consider a release branch
git checkout -b release/v1.0.0
git push -u origin release/v1.0.0
```

---

## Phase 2: VALIDATE

### Automated Checks

```bash
# Format check (should be clean)
black --check .
isort --check .

# Linting (no errors allowed)
ruff check .

# Type checking (should pass with no errors)
mypy src/ --ignore-missing-imports

# Security scan (no high/critical issues)
bandit -r src/ -ll

# Run all tests including slow ones
pytest tests/ -v --tb=short
pytest tests/smoke/ -v -m slow

# Reproducibility check
python scripts/sce_analysis.py --dataset sample --seed 42 --output run1
python scripts/sce_analysis.py --dataset sample --seed 42 --output run2
diff run1/data/metrics.json run2/data/metrics.json
```

### Manual Checks

- [ ] **Example scripts run**: `python examples/quick_start.py`
- [ ] **CLI works**: `python scripts/sce_analysis.py --help`
- [ ] **Config parsing works**: Try each dataset config
- [ ] **Output looks reasonable**: Spot-check reports and figures

---

## Phase 3: DOCUMENT

### Version Bump

```toml
# pyproject.toml
[project]
version = "1.0.0"  # Update this!
```

### CHANGELOG Update

```markdown
## [1.0.0] - 2026-01-15

### The Production Release

After 6 months of R&D, SCE is ready for production use. This release focuses 
on reliability, reproducibility, and ease of use.

### Highlights

- **Full reproducibility**: Same inputs → same outputs, guaranteed
- **Comprehensive testing**: 95% coverage, property tests, regression tests
- **Production-quality docs**: Real examples, honest limitations

### Added

- `StatisticalContextEngine.save()` and `.load()` for model persistence
- `--seed` CLI argument for reproducible runs
- Metadata capture in every report

### Changed

- Default XGBoost config now uses deterministic settings
- Correlation threshold increased to 0.85 (was 0.80)

### Fixed

- Float precision issues in sd_relative calculation (#42)
- Memory leak when processing >1M rows (#56)

### Breaking Changes

- `run_pipeline()` now requires explicit `seed` parameter
- Config format changed: `hierarchy_levels` is now a table, not a list
```

### README Updates

- [ ] Installation instructions are current
- [ ] Example code works with latest API
- [ ] Version badge shows correct version
- [ ] "What's New" section updated

---

## Phase 4: TEST INSTALL

### Clean Environment Test

```powershell
# Windows
python -m venv .release-test
.release-test\Scripts\activate
pip install -e ".[all]"

# Run quick validation
python -c "from statistical_context import StatisticalContextEngine; print('Import OK')"
python examples/quick_start.py
python scripts/sce_analysis.py --dataset sample --help
```

### Docker Test (If Applicable)

```bash
# Build fresh
docker build -t sce:release-test .

# Run smoke test
docker run sce:release-test python -c "import statistical_context; print('OK')"
```

### Cross-Platform Check

If possible, test on:
- [ ] Windows
- [ ] macOS
- [ ] Linux

At minimum, CI should cover Linux.

---

## Phase 5: TAG

### Tagging Convention

```bash
# Semantic versioning: MAJOR.MINOR.PATCH
# MAJOR: Breaking changes
# MINOR: New features (backward compatible)
# PATCH: Bug fixes (backward compatible)

# Create annotated tag
git tag -a v1.0.0 -m "Release v1.0.0 - The Production Release"

# Push tag
git push origin v1.0.0
```

### Pre-Release Tags

```bash
# For beta releases
git tag -a v1.0.0-beta.1 -m "Beta 1 for v1.0.0"

# For release candidates
git tag -a v1.0.0-rc.1 -m "Release candidate 1 for v1.0.0"
```

---

## Phase 6: PUBLISH

### PyPI Release

```bash
# Build distribution
python -m build

# Check the build
twine check dist/*

# Upload to TestPyPI first
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ statistical-context-engineering

# If everything works, upload to real PyPI
twine upload dist/*
```

### GitHub Release

1. Go to repository → Releases → "Draft a new release"
2. Choose the tag you just pushed
3. Title: `v1.0.0 - The Production Release`
4. Body: Copy from CHANGELOG
5. Attach artifacts if needed (built wheels, etc.)
6. Publish release

### Release Artifacts Checklist

- [ ] Source distribution (`.tar.gz`)
- [ ] Wheel (`.whl`)
- [ ] Example data (if included)
- [ ] Pre-trained model (if included)

---

## Phase 7: VERIFY

### Post-Release Verification

```bash
# Fresh environment
python -m venv .post-release-test
.post-release-test\Scripts\activate

# Install from PyPI (not local)
pip install statistical-context-engineering

# Verify it works
python -c "from statistical_context import StatisticalContextEngine; print(StatisticalContextEngine.__version__)"

# Run example
python -c "
from statistical_context import quick_demo
quick_demo()
"
```

### Documentation Site

- [ ] Docs are deployed and accessible
- [ ] Version selector shows new version
- [ ] Examples render correctly

### Announcement

- [ ] GitHub release is published
- [ ] Team/users notified (Slack, email, etc.)
- [ ] Social media if appropriate

---

## Release Checklist Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RELEASE CHECKLIST                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   PREPARATION                                                       │
│   ───────────                                                       │
│   [ ] Code freeze (no new features)                                │
│   [ ] All tests pass                                               │
│   [ ] Linting/formatting clean                                     │
│   [ ] Security scan clean                                          │
│                                                                     │
│   DOCUMENTATION                                                     │
│   ─────────────                                                     │
│   [ ] Version bumped in pyproject.toml                             │
│   [ ] CHANGELOG updated with release notes                         │
│   [ ] README current and accurate                                  │
│                                                                     │
│   TESTING                                                           │
│   ───────                                                           │
│   [ ] Clean install works                                          │
│   [ ] Examples run without error                                   │
│   [ ] CLI works                                                    │
│   [ ] Docker build works (if applicable)                           │
│                                                                     │
│   RELEASE                                                           │
│   ───────                                                           │
│   [ ] Git tag created and pushed                                   │
│   [ ] PyPI upload successful                                       │
│   [ ] GitHub release published                                     │
│                                                                     │
│   VERIFICATION                                                      │
│   ────────────                                                      │
│   [ ] Install from PyPI works                                      │
│   [ ] Docs are live                                                │
│   [ ] Team notified                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Rollback Plan

If something goes wrong:

### PyPI Rollback

```bash
# You can't delete versions from PyPI, but you can:
# 1. Yank the release (hides it but doesn't delete)
# 2. Release a patch version immediately

# Post-mortem:
# Document what happened
# Fix the issue
# Release v1.0.1
```

### Git Rollback

```bash
# Delete the tag locally
git tag -d v1.0.0

# Delete from remote
git push origin :refs/tags/v1.0.0

# Fix the issue and re-tag
```

---

## Post-Release Hygiene

### Immediately After Release

- [ ] Create next milestone on GitHub
- [ ] Open tracking issue for next version
- [ ] Merge any hotfix branches
- [ ] Update development version in `pyproject.toml` to next dev version

### Within a Week

- [ ] Review any issues filed against the release
- [ ] Plan hotfix if critical bugs found
- [ ] Retrospective: What could be smoother next time?

---

*"A release is not the end. It's a commitment to support what you shipped."*
