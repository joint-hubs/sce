# Documentation Style

> *"Documentation is a love letter that you write to your future self."*  
> — Damian Conway

---

## The Problem with Generic Docs

AI-generated documentation has tells:
- Phrases like "This module provides functionality for..."
- Perfect parallel structure everywhere
- No personality, no opinions, no voice
- Explains *what* the code does (obvious from reading it)
- Never explains *why* decisions were made

Your documentation should look like a **thoughtful human wrote it** — someone who cares about the reader and has opinions about the domain.

---

## The Documentation Hierarchy

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DOCUMENTATION LEVELS                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   README.md                                                         │
│   └── "What is this and why should I care?" (30 seconds to answer) │
│                                                                     │
│   EXAMPLES.md / Quick Start                                         │
│   └── "Show me it working" (5 minutes to run)                       │
│                                                                     │
│   User Guide / Tutorials                                            │
│   └── "Teach me the concepts" (30 minutes to read)                  │
│                                                                     │
│   API Reference                                                     │
│   └── "What are all the options?" (lookup as needed)                │
│                                                                     │
│   Architecture / Internals                                          │
│   └── "How does it actually work?" (for contributors)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## README.md: The First Impression

### The First Paragraph Rule

You have **one paragraph** to explain:
1. What this is
2. What problem it solves
3. Why someone should care

### ❌ Generic README Opening

```markdown
# Statistical Context Engineering

Statistical Context Engineering (SCE) is a Python library that provides 
functionality for computing statistical features and enriching datasets 
with contextual information for machine learning applications.
```

**Problems:**
- "provides functionality for" — meaningless filler
- No concrete problem statement
- Could describe any ML library

### ✅ Human-Written README Opening

```markdown
# Statistical Context Engineering

A $500K house in Manhattan means something completely different than $500K in 
rural Ohio. Raw prices tell you nothing without context.

**SCE enriches your features with market context** — group-level statistics 
that let your model understand where each data point sits relative to its peers. 
The result: 15-30% better predictions on regression tasks with hierarchical data.

```python
# Before: just raw features
df = pd.DataFrame({"city": ["NYC", "LA"], "price": [500000, 400000]})

# After: features + context (where does this price rank in its city?)
enriched = sce.enrich(df, hierarchy=["city"])
# Now includes: city_median, city_mean, city_percentile_rank, ...
```
```

**Why it works:**
- Opens with a concrete insight
- States the benefit (15-30% better predictions)
- Shows code immediately
- Has a voice (not robotic)

---

## Docstrings: Explain the Why

### Function Docstrings

```python
# ❌ GENERIC: Restates the function signature
def compute_group_statistics(df: pd.DataFrame, groupby: list[str]) -> dict:
    """
    Compute group statistics for the given dataframe.
    
    Args:
        df: The input dataframe.
        groupby: The columns to group by.
    
    Returns:
        A dictionary of statistics.
    """
```

```python
# ✅ HUMAN: Explains the insight and edge cases
def compute_group_statistics(df: pd.DataFrame, groupby: list[str]) -> dict:
    """
    Compute market positioning statistics for each group.
    
    The key insight: knowing that a house costs $500K means nothing in isolation.
    Knowing it's in the 90th percentile for its neighborhood tells you it's 
    expensive *relative to comparable properties*.
    
    We compute 9 statistics per group:
    - Location: min, q25, median, mean, q75, max
    - Spread: sd, sd_relative (sd/mean), mean_relative (value/group_mean)
    
    Note: Groups with fewer than 3 observations get NaN for standard deviation
    to avoid misleading precision. This is intentional — if you have 2 houses 
    in a zip code, you don't really know the "typical" price.
    
    Args:
        df: Must contain numeric column 'price' (or target column configured).
        groupby: Hierarchy levels to group by. Order matters: ["state", "city"] 
                 creates nested groups (cities within states).
    
    Returns:
        Nested dict: {group_key: {stat_name: value}}
    
    Raises:
        ValueError: If df is empty or groupby columns don't exist.
    
    Example:
        >>> stats = compute_group_statistics(listings, ["city"])
        >>> stats["NYC"]["median"]
        525000.0
    """
```

### Class Docstrings

```python
# ❌ GENERIC
class StatisticalContextEngine:
    """Engine for computing statistical context."""

# ✅ HUMAN
class StatisticalContextEngine:
    """
    The heart of SCE: computes and applies market context features.
    
    Design philosophy:
    - Fit on training data only (prevents data leakage)
    - Transform can be applied to any data (train, test, production)
    - Stateless after fitting (serialize and reuse)
    
    Typical workflow:
        engine = StatisticalContextEngine(hierarchy=["city", "property_type"])
        engine.fit(train_df)
        
        train_enriched = engine.transform(train_df)
        test_enriched = engine.transform(test_df)  # Uses training statistics!
    
    Why not just compute statistics on-the-fly?
        Leakage. If you compute median(test_df["city"]) you're using 
        test data to create features. The model will overfit to the test 
        distribution and fail in production.
    """
```

---

## Code Comments: The Art of Why

### Comment Philosophy

| Level | What to Comment | Example |
|-------|-----------------|---------|
| **Why** | Business logic, design decisions | "We cap outliers at 99th percentile because..." |
| **What (non-obvious)** | Complex algorithms | "Stable sort ensures reproducibility" |
| **Gotchas** | Edge cases, surprises | "XGBoost silently drops NaN — handle before training" |
| **References** | Links to docs, papers | "Algorithm from Smith et al. 2024" |

### ❌ Useless Comments

```python
# Loop through items
for item in items:
    # Process the item
    process(item)
    # Add to results
    results.append(item)
```

### ✅ Valuable Comments

```python
# We process cities before property types because city-level stats 
# are used to compute relative property type rankings within each city.
# Order matters here — don't parallelize without careful thought.
for level in ["city", "property_type"]:
    stats = compute_level_stats(df, level)
    
    # Merge stats back, using left join to preserve all rows.
    # Inner join would drop listings in rare categories.
    df = df.merge(stats, on=level, how="left")
```

---

## Config File Documentation

TOML/YAML configs should be self-documenting:

### ❌ Undocumented Config

```toml
[data]
path = "Datasets/airbnb/*.csv"

[dataset]
target_column = "price"
hierarchy_levels = ["city", "property_type"]
```

### ✅ Documented Config

```toml
# ═══════════════════════════════════════════════════════════════════
# Airbnb NYC Dataset Configuration
# 
# This dataset contains ~49K Airbnb listings from New York City.
# Interesting properties:
#   - Strong neighborhood effects (Manhattan vs Bronx pricing)
#   - Property type hierarchy (Entire home > Private room > Shared)
#   - Seasonal patterns (we don't model these yet)
# ═══════════════════════════════════════════════════════════════════

[data]
# Glob pattern allows multiple files. Currently just one, but we may 
# add historical data later for time-series experiments.
path = "Datasets/airbnb/*.csv"

[dataset]
name = "Airbnb NYC"
target_column = "price"  # Nightly rental price in USD

# Hierarchy levels for SCE feature computation.
# Order matters: city → neighborhood → property_type gives us nested context.
# Each listing gets stats relative to its city, its neighborhood within that 
# city, and its property type within that neighborhood.
[dataset.hierarchy_levels]
city = "neighbourhood_group"      # Borough (Manhattan, Brooklyn, etc.)
neighborhood = "neighbourhood"    # Specific neighborhood name
property_type = "room_type"       # Entire home, Private room, Shared room

[dataset.features]
# Categorical features (will be label-encoded)
categorical = ["neighbourhood_group", "room_type"]

# Numeric features (used as-is, plus generate SCE context features)
numeric = [
    "latitude", 
    "longitude",
    "minimum_nights",
    "number_of_reviews",
    "availability_365",
]

# Features to drop (identifiers, free text, redundant)
drop = ["id", "name", "host_name", "last_review"]

[run]
# XGBoost configs to try (from configs/models/xgboost.toml)
# Start with shallow trees, then try deeper if underfitting
xgboost_configs = ["shallow", "default", "regularized"]
```

---

## Error Messages

### ❌ Generic Errors

```python
raise ValueError("Invalid input")
raise RuntimeError("Something went wrong")
```

### ✅ Helpful Errors

```python
raise ValueError(
    f"Column '{target_col}' not found in dataframe. "
    f"Available columns: {list(df.columns)[:10]}... "
    f"Check your config file's 'target_column' setting."
)

raise RuntimeError(
    f"XGBoost training failed after {n_tries} attempts. "
    f"Last error: {last_error}. "
    f"Common causes: (1) NaN in features, (2) infinite values, "
    f"(3) mismatched feature count between train/test. "
    f"Run with --debug to see detailed diagnostics."
)
```

---

## CHANGELOG: Tell the Story

### ❌ Useless Changelog

```markdown
## v1.2.0
- Updated dependencies
- Fixed bugs
- Added new features
```

### ✅ Narrative Changelog

```markdown
## v1.2.0 (2026-01-15) — The Reproducibility Release

This release focuses on making experiments fully reproducible. After debugging 
a customer issue where "the model gave different results on their server," we 
discovered several sources of non-determinism. All fixed now.

### Breaking Changes

- `run_pipeline()` now **requires** a `seed` parameter. No more implicit randomness.
  
  ```python
  # Before (non-deterministic)
  result = run_pipeline(config)
  
  # After (reproducible)
  result = run_pipeline(config, seed=42)
  ```

### Added

- **Metadata capture**: Every run now saves `metadata.json` with git SHA, 
  package versions, and input data hash. You can always recreate a run.
  
- **Reproducibility test**: New `pytest --reproduce` flag runs the pipeline 
  twice and verifies identical outputs.

### Fixed

- **Dict ordering bug**: Feature names were processed in arbitrary order on 
  Python <3.7. Now explicitly sorted. (#142)
  
- **XGBoost GPU non-determinism**: Set `sampling_method="uniform"` by default 
  when using GPU. Slight speed penalty but reproducible results. (#156)

### Known Issues

- Multi-threaded feature computation is still non-deterministic. Use 
  `--single-thread` for exact reproducibility (5x slower).
```

---

## The Voice Test

Read your documentation aloud. Ask:

1. **Does it sound like a person talking?** Or a corporate manual?
2. **Would I understand this if I knew nothing about the project?**
3. **Is there any personality?** Humor, opinions, honest admissions?
4. **Are the examples real?** Or generic placeholders?

### Voice Markers to Use

| Pattern | Example |
|---------|---------|
| Direct address | "You'll want to..." not "Users should..." |
| Honest limitations | "This is slow but simple" |
| Opinions | "We recommend X because..." |
| Casual contractions | "It's" not "It is", "don't" not "do not" |
| Questions | "Why not just compute on-the-fly?" |

### Voice Markers to Avoid

| Pattern | Why |
|---------|-----|
| "This module provides..." | Robotic, says nothing |
| "It should be noted that..." | Wordy, passive |
| "The user should..." | Distancing, impersonal |
| "Various" / "Numerous" | Vague, unhelpful |
| "Functionality" | Corporate buzzword |

---

## Documentation Checklist

### README

- [ ] First paragraph answers "what and why"
- [ ] Working code example in first 30 lines
- [ ] Installation is copy-paste-able
- [ ] Has a "Quick Start" section
- [ ] Mentions known limitations honestly

### Docstrings

- [ ] Explain the *why*, not just the *what*
- [ ] Include realistic examples
- [ ] Document edge cases and gotchas
- [ ] Link to relevant concepts

### Configs

- [ ] Every non-obvious setting is commented
- [ ] Show example values, not just types
- [ ] Explain when to change defaults

### CHANGELOG

- [ ] Tells a story, not a list
- [ ] Breaking changes are highlighted
- [ ] Includes migration examples
- [ ] Links to relevant issues/PRs

---

*"The best documentation doesn't just tell you how to use something — it makes you feel like the author is sitting next to you, explaining their thinking."*
