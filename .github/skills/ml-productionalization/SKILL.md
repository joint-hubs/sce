---
name: ml-productionalization
description: |
  Productionalize R&D ML projects with the craft of an experienced engineer. 
  Use when: "wrap up my ML project", "make this production ready", "productionalize",
  "I need to ship this model", "prepare for release", "add tests to my pipeline",
  "make my code look professional", "this needs to not look AI-generated",
  "review my ML code", "make this reproducible".
  
  Covers testing, reproducibility, documentation, code quality, and release prep.
metadata:
  author: fenix
  version: "1.0"
  domain: machine-learning
---

# ML Productionalization

> *"R&D got you here. Engineering gets you to production."*

## Purpose

You're a data scientist who built something that works. Now you need to ship it. This skill transforms experimental code into production-grade software that:

- **Runs the same way every time** (reproducibility)
- **Catches problems before users do** (testing)
- **Explains itself to future maintainers** (documentation)
- **Looks like a human crafted it** (code quality)

## Index

📋 **[INDEX.md](INDEX.md)** — Full file listing and creation status

## Philosophy First

Before touching code, read **[THE_CRAFT.md](THE_CRAFT.md)**. It explains:
- Why code should tell a story
- How to add human fingerprints
- The difference between "works" and "ships"

## The Productionalization Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                 R&D → PRODUCTION WORKFLOW                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐      │
│   │  AUDIT  │ ──► │  TEST   │ ──► │  DOCS   │ ──► │ RELEASE │      │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘      │
│       │               │               │               │             │
│       ▼               ▼               ▼               ▼             │
│   CODE_REVIEW    TESTING_      DOCUMENTATION_   RELEASE_           │
│   .md            WORKFLOW.md   STYLE.md         CHECKLIST.md       │
│                                                                     │
│   + REPRODUCIBILITY.md (cross-cutting concern)                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Phase 1: AUDIT — Know What You Have

Before adding anything, understand the current state.

### Run the Validation Script

```bash
python .github/skills/ml-productionalization/scripts/validate_project.py --path .
```

This checks:
- [ ] Dependencies are pinned (`requirements.txt` or `pyproject.toml`)
- [ ] Entry points are documented
- [ ] Random seeds are set
- [ ] Config files are complete
- [ ] No hardcoded paths

### Self-Review Checklist

Read **[CODE_REVIEW.md](CODE_REVIEW.md)** and answer honestly:
- Does this code embarrass me?
- Could a new team member understand it in 30 minutes?
- Would I be comfortable debugging this at 2am?

## Phase 2: TEST — Prove It Works

ML testing is different from web app testing. We care about:

| Test Type | What It Proves |
|-----------|----------------|
| **Unit tests** | Individual functions work correctly |
| **Property tests** | Statistical guarantees hold |
| **Integration tests** | Pipeline stages connect properly |
| **Regression tests** | Changes don't break existing behavior |
| **Smoke tests** | The whole thing runs end-to-end |

📋 **[TESTING_WORKFLOW.md](TESTING_WORKFLOW.md)** — Detailed testing strategy

### Minimum Test Coverage for ML Projects

```python
# 1. Data loading doesn't explode
def test_load_dataset_returns_dataframe():
    df = load_dataset("configs/datasets/sample.toml")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

# 2. Feature engineering is deterministic
def test_feature_engineering_is_reproducible():
    df1 = compute_features(data, seed=42)
    df2 = compute_features(data, seed=42)
    pd.testing.assert_frame_equal(df1, df2)

# 3. Model training produces expected outputs
def test_model_outputs_reasonable_predictions():
    model = train_model(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.min() >= 0  # prices can't be negative
    assert preds.max() < 1e9  # sanity check
```

## Phase 3: DOCS — Tell the Story

Generic documentation fails. Good documentation has:

- **Personality** — written by a human, not a template
- **Narrative flow** — reads like a story, not a reference
- **Real examples** — actual inputs and outputs, not placeholders
- **Honest limitations** — what doesn't work, and why

📋 **[DOCUMENTATION_STYLE.md](DOCUMENTATION_STYLE.md)** — Writing docs that don't look AI-generated

### Documentation Checklist

- [ ] README explains the "what" and "why" in the first paragraph
- [ ] CONTRIBUTING shows how to run tests locally
- [ ] Config examples are complete and commented
- [ ] API docs have real-world usage examples
- [ ] Known issues are documented, not hidden

## Phase 4: REPRODUCIBILITY — Same Results, Every Time

The #1 cause of ML project failure: "it worked on my machine."

📋 **[REPRODUCIBILITY.md](REPRODUCIBILITY.md)** — Making experiments reproducible

### The Reproducibility Trinity

```python
# 1. Pin everything
# requirements.txt (generated from pip freeze)
numpy==1.24.3
pandas==2.0.3
xgboost==2.0.0

# 2. Seed everything
import numpy as np
import random
np.random.seed(42)
random.seed(42)

# 3. Log everything
metadata = {
    "timestamp": datetime.now().isoformat(),
    "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip(),
    "config_hash": hashlib.md5(config_string.encode()).hexdigest(),
    "python_version": sys.version,
}
```

## Phase 5: RELEASE — Ship It

📋 **[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)** — Final pre-release gates

### Pre-Release Validation

```bash
# Clean install test
python -m venv .test-env
.test-env/Scripts/activate  # Windows
pip install -e ".[all]"
python -m pytest tests/ -v

# Smoke test the main entry point
python scripts/sce_analysis.py --dataset sample --dry-run
```

### Release Artifacts

- [ ] Version bump in `pyproject.toml`
- [ ] CHANGELOG updated with notable changes
- [ ] Git tag created (`git tag v1.0.0`)
- [ ] Package builds successfully (`python -m build`)

## Quick Reference

| I need to... | Read this |
|--------------|-----------|
| Understand the philosophy | [THE_CRAFT.md](THE_CRAFT.md) |
| Add tests | [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md) |
| Make it reproducible | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Write better docs | [DOCUMENTATION_STYLE.md](DOCUMENTATION_STYLE.md) |
| Review my code | [CODE_REVIEW.md](CODE_REVIEW.md) |
| Prepare release | [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) |
| Run automated checks | [scripts/validate_project.py](scripts/validate_project.py) |

## The Mindset Shift

| R&D Thinking | Production Thinking |
|--------------|---------------------|
| "It works!" | "It works reliably, every time, on any machine" |
| "I know what this does" | "Anyone can understand what this does" |
| "I'll fix it later" | "This is the version that ships" |
| "The model is the product" | "The system is the product" |

---

*"Shipping is a feature. Everything else is just a demo."*
