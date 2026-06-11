# Reproducibility

> *"If you can't reproduce it, you didn't discover it. You stumbled onto it."*

---

## Why Reproducibility Matters

ML projects die when:
- "It worked yesterday but not today"
- "It works on my machine but not in production"
- "I can't recreate the model from 3 months ago"
- "We got different results with the same config"

Reproducibility isn't optional. It's survival.

---

## The Three Pillars

```
┌─────────────────────────────────────────────────────────────────────┐
│                    REPRODUCIBILITY PILLARS                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│     ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│     │             │    │             │    │             │          │
│     │     PIN     │    │    SEED     │    │     LOG     │          │
│     │ EVERYTHING  │    │ EVERYTHING  │    │ EVERYTHING  │          │
│     │             │    │             │    │             │          │
│     └─────────────┘    └─────────────┘    └─────────────┘          │
│                                                                     │
│     Dependencies       Randomness          Metadata                 │
│     ───────────        ──────────          ────────                │
│     • Python ver       • numpy.seed        • Git SHA               │
│     • Package vers     • random.seed       • Config hash           │
│     • System deps      • XGBoost seed      • Timestamps            │
│     • CUDA version     • Train/test split  • Input file hashes     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Pillar 1: Pin Everything

### Python Dependencies

```toml
# pyproject.toml — use specific versions, not ranges

[project]
dependencies = [
    "pandas==2.0.3",      # Not "pandas>=2.0"
    "numpy==1.24.3",
    "xgboost==2.0.0",
    "scikit-learn==1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest==7.4.0",
    "hypothesis==6.82.0",
]
```

### Generate a Lock File

```bash
# Create reproducible environment
pip freeze > requirements-lock.txt

# Recreate environment exactly
pip install -r requirements-lock.txt
```

### System Dependencies

```dockerfile
# Dockerfile — pin the base image

FROM python:3.11.4-slim  # Not python:3.11 or python:latest

# Pin system packages too
RUN apt-get update && apt-get install -y \
    libgomp1=12.2.0-14 \
    && rm -rf /var/lib/apt/lists/*
```

### CUDA/GPU Versions

```yaml
# environment.yml for conda

dependencies:
  - python=3.11.4
  - cudatoolkit=11.8.0
  - pytorch=2.0.1=py3.11_cuda11.8_cudnn8.7.0_0
```

---

## Pillar 2: Seed Everything

### The Seed Propagation Pattern

```python
# sce/reproducibility.py

import random
import numpy as np
import os

def set_global_seed(seed: int = 42) -> None:
    """
    Set all random seeds for reproducibility.
    
    Call this ONCE at the start of every script/notebook.
    
    Note: This doesn't guarantee 100% reproducibility with GPU operations
    or multi-threading. For that, see set_deterministic_mode().
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # Framework-specific seeds
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def set_deterministic_mode() -> None:
    """
    Enable fully deterministic operations.
    
    WARNING: This can significantly slow down GPU operations.
    Use only when exact reproducibility is required.
    """
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    try:
        import torch
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    except ImportError:
        pass
```

### XGBoost Specific Seeds

```python
# XGBoost has its own random state parameter

model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=5,
    random_state=42,  # Controls feature sampling, tree building
    # For GPU training, also set:
    sampling_method="uniform",  # Not "gradient_based" which is non-deterministic
)
```

### Train/Test Split Seeds

```python
from sklearn.model_selection import train_test_split

# ALWAYS specify random_state
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,  # Make split reproducible
    stratify=y_categorical if is_classification else None,
)
```

### Sampling Seeds

```python
def sample_combinations(df: pd.DataFrame, pct: float, seed: int = 42) -> pd.DataFrame:
    """
    Sample a percentage of rows.
    
    Note: Using df.sample(frac=pct) without random_state is a bug.
    """
    return df.sample(frac=pct, random_state=seed)
```

---

## Pillar 3: Log Everything

### Run Metadata

```python
# sce/metadata.py

import subprocess
import hashlib
import platform
import json
from datetime import datetime
from pathlib import Path

def capture_run_metadata(config_path: Path, data_path: Path) -> dict:
    """
    Capture everything needed to reproduce this run.
    
    Store this in reports/{name}/data/metadata.json
    """
    return {
        # When
        "timestamp": datetime.utcnow().isoformat() + "Z",
        
        # What code
        "git_sha": get_git_sha(),
        "git_dirty": is_git_dirty(),
        "git_branch": get_git_branch(),
        
        # What config
        "config_path": str(config_path),
        "config_hash": file_hash(config_path),
        
        # What data
        "data_path": str(data_path),
        "data_hash": file_hash(data_path),
        "data_rows": count_rows(data_path),
        
        # What environment
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": get_installed_packages(),
        
        # What hardware
        "cpu_count": os.cpu_count(),
        "gpu_available": check_gpu_available(),
    }


def get_git_sha() -> str:
    """Get current git commit SHA."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except subprocess.CalledProcessError:
        return "unknown"


def is_git_dirty() -> bool:
    """Check if there are uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True
        )
        return bool(result.stdout.strip())
    except subprocess.CalledProcessError:
        return True  # Assume dirty if can't check


def file_hash(path: Path) -> str:
    """Compute MD5 hash of a file."""
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_installed_packages() -> dict[str, str]:
    """Get versions of key packages."""
    packages = {}
    for pkg in ["pandas", "numpy", "xgboost", "scikit-learn"]:
        try:
            import importlib.metadata
            packages[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            packages[pkg] = "not installed"
    return packages
```

### Example metadata.json

```json
{
  "timestamp": "2026-01-15T14:32:18Z",
  "git_sha": "a1b2c3d4e5f6789",
  "git_dirty": false,
  "git_branch": "main",
  "config_path": "configs/datasets/airbnb.toml",
  "config_hash": "abc123def456",
  "data_path": "Datasets/airbnb/listings.csv",
  "data_hash": "789xyz",
  "data_rows": 48895,
  "python_version": "3.11.4",
  "platform": "Windows-10-10.0.22631-SP0",
  "packages": {
    "pandas": "2.0.3",
    "numpy": "1.24.3",
    "xgboost": "2.0.0",
    "scikit-learn": "1.3.0"
  },
  "cpu_count": 8,
  "gpu_available": true
}
```

### Save Model Artifacts

```python
def save_trained_model(model, output_dir: Path, metadata: dict) -> None:
    """
    Save model with full reproducibility information.
    
    Creates:
        output_dir/
        ├── model.json           # XGBoost model
        ├── metadata.json        # Run information
        ├── feature_names.json   # Ordered feature list
        └── training_params.json # Hyperparameters used
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model itself
    model.save_model(output_dir / "model.json")
    
    # Metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Feature names in order (critical for prediction)
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(model.get_booster().feature_names, f)
    
    # Training parameters
    with open(output_dir / "training_params.json", "w") as f:
        json.dump(model.get_params(), f, indent=2)
```

---

## The Reproducibility Contract

```python
# At the start of every script

def main():
    # 1. Set seeds FIRST, before any imports that use randomness
    from statistical_context.reproducibility import set_global_seed
    set_global_seed(42)
    
    # 2. Capture metadata BEFORE doing work
    from statistical_context.metadata import capture_run_metadata
    metadata = capture_run_metadata(args.config, args.data)
    
    # 3. Do the actual work
    result = run_pipeline(args)
    
    # 4. Save metadata WITH the results
    save_results(result, metadata, args.output_dir)
```

---

## Verifying Reproducibility

### The Reproducibility Test

```python
# tests/reproducibility/test_determinism.py

def test_same_seed_same_result():
    """Running twice with the same seed should produce identical results."""
    from statistical_context.reproducibility import set_global_seed
    
    set_global_seed(42)
    result1 = run_pipeline(config, data)
    
    set_global_seed(42)
    result2 = run_pipeline(config, data)
    
    # RMSE should be exactly equal
    assert result1["rmse"] == result2["rmse"]
    
    # Predictions should be identical
    np.testing.assert_array_equal(result1["predictions"], result2["predictions"])


def test_different_seed_different_result():
    """Different seeds should produce (slightly) different results."""
    set_global_seed(42)
    result1 = run_pipeline(config, data)
    
    set_global_seed(43)
    result2 = run_pipeline(config, data)
    
    # Results should differ (proves randomness is actually being used)
    assert result1["rmse"] != result2["rmse"]
```

### CI Reproducibility Check

```yaml
# .github/workflows/reproducibility.yml

name: Reproducibility Check
on: [push]

jobs:
  reproduce:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11.4'  # Pin minor version
      
      - name: Install exact dependencies
        run: pip install -r requirements-lock.txt
      
      - name: Run pipeline twice
        run: |
          python scripts/sce_analysis.py --dataset sample --seed 42 --output-dir run1
          python scripts/sce_analysis.py --dataset sample --seed 42 --output-dir run2
      
      - name: Compare outputs
        run: |
          diff run1/data/metrics.json run2/data/metrics.json
          diff run1/data/predictions.csv run2/data/predictions.csv
```

---

## Common Reproducibility Killers

### 1. Floating Point Non-Determinism

```python
# PROBLEM: Order of operations affects floating point
sum1 = a + b + c
sum2 = c + b + a  # Might be different!

# SOLUTION: Use stable algorithms
from math import fsum
total = fsum([a, b, c])  # Order-independent
```

### 2. Dictionary Ordering

```python
# PROBLEM: Dict ordering wasn't guaranteed before Python 3.7
# STILL problematic when iterating for model features

# SOLUTION: Sort explicitly
features = sorted(feature_dict.keys())
for feature in features:
    process(feature_dict[feature])
```

### 3. File System Ordering

```python
# PROBLEM: glob order is OS-dependent
files = glob.glob("data/*.csv")  # Different order on Windows vs Linux

# SOLUTION: Sort the results
files = sorted(glob.glob("data/*.csv"))
```

### 4. Parallel Execution

```python
# PROBLEM: Thread order is non-deterministic
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process, items))  # Order varies!

# SOLUTION: Use index-based reassembly
results = [None] * len(items)
with ThreadPoolExecutor() as executor:
    futures = {executor.submit(process, item): i for i, item in enumerate(items)}
    for future in as_completed(futures):
        results[futures[future]] = future.result()
```

---

## Reproducibility Checklist

### Code Review

- [ ] All random operations have explicit seeds
- [ ] No global state that varies between runs
- [ ] File operations use sorted ordering
- [ ] Metadata is captured before processing

### Environment

- [ ] Python version is pinned (3.11.4, not 3.11)
- [ ] All packages have pinned versions
- [ ] requirements-lock.txt or lock file exists
- [ ] Docker/conda environment is versioned

### Artifacts

- [ ] metadata.json saved with every run
- [ ] Git SHA recorded
- [ ] Input data hash recorded
- [ ] Model artifacts include training params

### Verification

- [ ] Same-seed test passes
- [ ] CI reproducibility check enabled
- [ ] README documents how to reproduce

---

*"Reproducibility is not a feature. It's the difference between science and alchemy."*
