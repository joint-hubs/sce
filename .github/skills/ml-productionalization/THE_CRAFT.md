# The Craft

> *"Any fool can write code that a computer can understand. Good programmers write code that humans can understand."* — Martin Fowler

---

## Why This Document Exists

AI can write code that works. But AI-written code has a tell: it's *too clean*, *too generic*, *too consistent*. It lacks the fingerprints of someone who actually thought about the problem.

This document teaches you to write code that looks like a brilliant human wrote it.

---

## The Story Principle

Code should read like a story, not a recipe.

### ❌ Recipe Code (AI-typical)

```python
def process_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Process the dataframe according to configuration."""
    # Apply transformations
    for transform in config["transforms"]:
        df = apply_transform(df, transform)
    
    # Filter data
    df = df[df["valid"] == True]
    
    # Return processed data
    return df
```

### ✅ Story Code (Human-crafted)

```python
def enrich_with_market_context(listings: pd.DataFrame, hierarchy: HierarchyConfig) -> pd.DataFrame:
    """
    The insight here is subtle but powerful: a $500K house in Manhattan 
    means something completely different than $500K in rural Ohio.
    
    We compute group-level statistics (medians, spreads) that let the model
    understand where each listing sits within its local market context.
    """
    # First pass: compute what "normal" looks like for each market segment
    market_stats = compute_segment_statistics(listings, hierarchy)
    
    # Now each listing gets a "compared to my neighbors" score
    enriched = join_context_features(listings, market_stats)
    
    # Sanity check: we should have strictly more columns, same rows
    assert len(enriched.columns) > len(listings.columns), "Context features missing"
    assert len(enriched) == len(listings), "Rows changed during enrichment"
    
    return enriched
```

### What Makes Story Code Different

| Aspect | Recipe | Story |
|--------|--------|-------|
| **Naming** | `process_data`, `config` | `enrich_with_market_context`, `hierarchy` |
| **Docstring** | States the obvious | Explains the *insight* |
| **Comments** | Repeat the code | Explain the *why* |
| **Assertions** | Missing | Guard the invariants |

---

## Human Fingerprints

Real code written by experienced engineers has quirks. Here's how to add them authentically:

### 1. Opinionated Comments

```python
# BAD: Generic
# Check if the value is valid
if value > 0:

# GOOD: Opinionated
# Negative prices would mean someone's paying us to take the house. 
# That's either a data error or a very haunted property.
if value > 0:
```

### 2. Honest Limitations

```python
def sample_feature_combinations(df: pd.DataFrame, sample_pct: float = 0.05) -> pd.DataFrame:
    """
    We sample combinations rather than computing all of them because:
    1. Memory would explode (2^n categorical combinations)
    2. Most combinations have zero data anyway
    3. 5% random sample gives us 95% of the predictive signal
    
    Limitation: This makes results non-deterministic unless you set a seed.
    We chose to accept this tradeoff for 10x speed improvement.
    """
```

### 3. Pragmatic Shortcuts

```python
# Sometimes the "correct" solution is overkill
def quick_and_dirty_dedup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yes, there are fancier dedup strategies. 
    No, they don't matter for this use case.
    We just drop exact duplicates and move on.
    """
    return df.drop_duplicates()
```

### 4. Future Warnings

```python
def compute_statistics(df: pd.DataFrame, groupby_cols: list[str]) -> dict:
    # TODO(2026-Q2): This gets slow beyond 10M rows. 
    # If we hit that scale, switch to Polars or push to BigQuery.
    # For now, pandas is fine and the code is simpler.
```

### 5. Version Comments

```python
# v1: Tried median absolute deviation — too slow, abandoned
# v2: Tried rolling statistics — made leakage bugs too easy
# v3 (current): Group-level aggregates computed once on training data
#               Simple, fast, leakage-proof
```

---

## Naming That Tells a Story

### Variables

```python
# BAD: What is 'df2'?
df2 = df.merge(stats, on="city")

# GOOD: I know exactly what this is
listings_with_city_context = listings.merge(city_statistics, on="city")
```

### Functions

```python
# BAD: Generic verbs
def process() -> None
def handle_data() -> pd.DataFrame
def do_analysis() -> dict

# GOOD: Specific actions
def strip_outliers_by_iqr() -> pd.DataFrame
def train_baseline_model() -> XGBRegressor
def generate_feature_importance_report() -> Path
```

### Classes

```python
# BAD: Manager/Handler/Processor antipatterns
class DataProcessor:
class ModelHandler:
class FeatureManager:

# GOOD: Domain concepts
class StatisticalContextEngine:
class HierarchyLevel:
class FeatureSelectionPipeline:
```

---

## Comment Philosophy

### Don't Comment What, Comment Why

```python
# BAD: Restates the code
# Increment counter by one
counter += 1

# GOOD: Explains intent
# Track how many rows had missing target values for the audit report
missing_target_count += 1
```

### Comment at the Right Level

```python
# BAD: Too granular
# Create empty list
results = []
# Loop through items
for item in items:
    # Append to results
    results.append(transform(item))

# GOOD: Block-level intent
# Transform each raw record into model-ready features,
# preserving order for later alignment with prediction outputs
results = [transform(item) for item in items]
```

### Comment the Non-Obvious

```python
# This looks like a bug, but it's intentional:
# XGBoost's feature_importances_ sums to 1.0, but only when
# using 'weight' importance type. 'gain' type doesn't normalize.
# We renormalize here for consistent downstream interpretation.
importance = importance / importance.sum()
```

---

## Structure That Guides

### Logical File Organization

```python
# File: statistical_context.py

# ═══════════════════════════════════════════════════════════════════
# CORE ENGINE
# ═══════════════════════════════════════════════════════════════════

class StatisticalContextEngine:
    """The heart of the SCE system."""
    ...

# ═══════════════════════════════════════════════════════════════════
# STATISTICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════

def compute_group_statistics(df: pd.DataFrame, groupby: list[str]) -> dict:
    ...

def aggregate_statistics(stats: list[dict]) -> dict:
    ...

# ═══════════════════════════════════════════════════════════════════
# FEATURE ENRICHMENT
# ═══════════════════════════════════════════════════════════════════

def enrich_with_context(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    ...
```

### Function Order

Put functions in the order they're called:

```python
# 1. Entry point
def run_pipeline(config: Config) -> Report:
    data = load_data(config)
    features = engineer_features(data)
    model = train_model(features)
    return evaluate_model(model)

# 2. Then the functions, in call order
def load_data(config: Config) -> pd.DataFrame:
    ...

def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    ...

def train_model(features: pd.DataFrame) -> Model:
    ...

def evaluate_model(model: Model) -> Report:
    ...
```

---

## The Craft Checklist

Before submitting any code, ask:

- [ ] **Would I be proud to show this to a senior engineer?**
- [ ] **Can I explain the "why" for every major decision?**
- [ ] **Are variable names self-documenting?**
- [ ] **Do comments add value, not noise?**
- [ ] **Is there a logical flow from top to bottom?**
- [ ] **Did I acknowledge limitations honestly?**
- [ ] **Does this look like a human wrote it?**

---

## Anti-Patterns to Avoid

### AI-Generated Tells

| Pattern | Why It's a Tell | Fix |
|---------|-----------------|-----|
| Perfect consistency | Real code has personality | Add opinions, shortcuts |
| Generic docstrings | "Process the data" says nothing | Explain the insight |
| Over-abstraction | 3 classes where 1 function works | Be pragmatic |
| No TODOs or FIXMEs | Real code has rough edges | Be honest |
| Parallel structure everywhere | Too symmetrical | Let asymmetry exist |

### Cargo Cult Engineering

```python
# BAD: Pattern without understanding
class DataProcessorFactory:  # Why a factory? For one implementation?
    def create_processor(self) -> DataProcessor:
        return DataProcessor()

# GOOD: Just do the thing
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

---

## The Mindset

You're not writing code for the computer. The computer will understand anything syntactically correct.

You're writing code for:
1. **Future you** (3 months from now, debugging at midnight)
2. **Your teammates** (who need to modify this next sprint)
3. **The maintainer** (who inherits this when you move on)

Write code you'd want to inherit.

---

*"The craft is in the caring. Caring about the reader. Caring about the edge cases. Caring about the story the code tells."*
