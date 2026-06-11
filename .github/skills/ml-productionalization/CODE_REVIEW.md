# Code Review Checklist

> *"Would I be proud to show this to the best engineer I know?"*

---

## Why Self-Review Matters

Before anyone else sees your code, you should be its harshest critic. Every issue you catch yourself is one less round-trip with reviewers — and one less embarrassing moment.

This checklist is designed for **ML productionalization** — taking R&D code to production quality.

---

## The 30-Second Gut Check

Before going through the detailed checklist, answer honestly:

| Question | If "No"... |
|----------|------------|
| Would I want to debug this at 2am? | Simplify it |
| Could a new team member understand this in 30 minutes? | Add context |
| Am I proud of this code? | Refactor before submitting |
| Does this look like a human wrote it? | Add personality |

---

## Code Quality

### Naming

- [ ] **Variables tell a story**: Not `df2`, but `listings_with_city_context`
- [ ] **Functions are verbs**: Not `data()`, but `load_training_data()`
- [ ] **Classes are nouns**: Not `Processor`, but `StatisticalContextEngine`
- [ ] **No abbreviations that aren't universal**: `num` is fine, `lst_prcs` is not
- [ ] **Consistent naming**: If one function uses `df`, don't switch to `dataframe` elsewhere

### Structure

- [ ] **Functions are <50 lines**: If longer, extract subfunctions
- [ ] **Single responsibility**: Each function does ONE thing
- [ ] **Logical ordering**: Functions appear in the order they're called
- [ ] **No deep nesting**: Max 3 levels of indentation
- [ ] **Related code is grouped**: With clear section headers if needed

### Comments

- [ ] **Explain why, not what**: The code shows what; comments show intent
- [ ] **No stale comments**: If code changed, comment changed too
- [ ] **Complex logic is documented**: Algorithms, edge cases, magic numbers
- [ ] **TODO comments have context**: `TODO(2026-Q2): Optimize for large datasets`

---

## ML-Specific Checks

### Data Leakage Prevention

- [ ] **Train/test split before any preprocessing**
  ```python
  # RIGHT: Split first
  X_train, X_test = train_test_split(X)
  scaler.fit(X_train)
  
  # WRONG: Fit on all data
  scaler.fit(X)
  X_train, X_test = train_test_split(X)
  ```

- [ ] **SCE statistics computed on training data only**
  ```python
  # RIGHT
  engine.fit(train_df)
  train_enriched = engine.transform(train_df)
  test_enriched = engine.transform(test_df)
  
  # WRONG
  all_enriched = engine.fit_transform(pd.concat([train_df, test_df]))
  ```

- [ ] **No future data in time-series**: If data has timestamps, verify ordering

### Reproducibility

- [ ] **Random seeds are explicit**: Every `np.random`, `random`, `train_test_split`
- [ ] **Seeds are configurable**: Not hardcoded `42` everywhere
- [ ] **Metadata is logged**: Git SHA, config hash, timestamps

### Numerical Stability

- [ ] **No division by zero**: Check denominators, use `np.divide(..., where=...)`
- [ ] **NaN handling is explicit**: Don't rely on implicit behavior
- [ ] **Outlier handling is documented**: If you cap values, explain threshold choice
- [ ] **Floating point comparisons use tolerance**: `np.isclose()`, not `==`

### Performance

- [ ] **No unnecessary copies**: Use `df["col"]` not `df[["col"]].copy()`
- [ ] **Vectorized operations**: No Python loops over DataFrame rows
- [ ] **Sampling for large datasets**: `df.sample(frac=0.05)` during development
- [ ] **Progress bars for long operations**: `tqdm` for loops >1 minute

---

## Error Handling

### Fail Fast, Fail Clear

- [ ] **Validate inputs early**
  ```python
  def process(df: pd.DataFrame, target: str) -> pd.DataFrame:
      if df.empty:
          raise ValueError("Input dataframe is empty")
      if target not in df.columns:
          raise ValueError(f"Target '{target}' not in columns: {list(df.columns)}")
  ```

- [ ] **Error messages are actionable**
  ```python
  # BAD
  raise ValueError("Invalid config")
  
  # GOOD
  raise ValueError(
      f"Config key 'hierarchy_levels' missing. "
      f"Add it to {config_path} or see configs/README.md for examples."
  )
  ```

- [ ] **Warnings for recoverable issues**
  ```python
  import warnings
  if missing_pct > 0.1:
      warnings.warn(
          f"{missing_pct:.1%} of rows have missing target. "
          f"These will be dropped. Consider investigating data quality."
      )
  ```

---

## Testing

### Coverage

- [ ] **Happy path is tested**: Normal inputs produce expected outputs
- [ ] **Edge cases are tested**: Empty inputs, single values, extreme values
- [ ] **Error cases are tested**: Verify exceptions are raised correctly
- [ ] **Regression tests exist**: Known-good outputs are verified

### Quality

- [ ] **Tests document behavior**: Reading tests teaches you the API
- [ ] **Tests are deterministic**: Same result every run (use seeds!)
- [ ] **Tests are fast**: <10 seconds for the unit test suite
- [ ] **Tests are isolated**: No shared state between tests

---

## Documentation

### Code Documentation

- [ ] **Every public function has a docstring**
- [ ] **Docstrings have examples**: Real inputs and outputs
- [ ] **Complex logic has inline comments**: Especially algorithms and edge cases
- [ ] **Magic numbers are explained**: Why `0.8` correlation threshold?

### Project Documentation

- [ ] **README is current**: Reflects the actual code state
- [ ] **Config examples work**: Copy-paste-able
- [ ] **API changes are documented**: In CHANGELOG or version notes

---

## Production Readiness

### Robustness

- [ ] **Handles missing values gracefully**
- [ ] **Handles unexpected categories**: Categories in test that weren't in train
- [ ] **Has sensible defaults**: Works out-of-box for common cases
- [ ] **Timeouts on external calls**: API calls, database queries
- [ ] **Workflow dependencies are declared**: Scripts run in CI/Actions have their packages listed in the extras that CI installs

### Observability

- [ ] **Logging at appropriate levels**: `INFO` for milestones, `DEBUG` for details
- [ ] **Metrics are exposed**: Training time, prediction latency, data shapes
- [ ] **Errors are traceable**: Include request IDs, timestamps, input hashes

### Security

- [ ] **No secrets in code**: Use environment variables
- [ ] **No PII in logs**: Hash or redact sensitive data
- [ ] **Input validation**: Don't trust external data

---

## The Anti-Pattern Detector

Ask yourself if your code has these smells:

| Smell | Fix |
|-------|-----|
| **God class** (does everything) | Split by responsibility |
| **Shotgun surgery** (one change touches many files) | Better abstraction |
| **Feature envy** (uses other class's data constantly) | Move method to that class |
| **Primitive obsession** (dicts everywhere) | Create domain classes |
| **Comments explaining bad code** | Make the code self-explanatory |
| **Dead code** (unused functions/imports) | Delete it |
| **Copy-paste** (duplicated logic) | Extract to function |

---

## Pre-Commit Checklist

Run before every commit:

```bash
# Formatting
black .
isort .

# Linting
ruff check .
mypy src/

# Tests
pytest tests/unit/ -v

# Security
bandit -r src/
```

---

## The Final Question

> *"If the person who will maintain this code in 2 years is a violent psychopath who knows where I live... is this code clear enough to keep me safe?"*

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PRE-SUBMIT CHECKLIST                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   BASICS                          ML-SPECIFIC                       │
│   ──────                          ───────────                       │
│   [ ] Names tell a story          [ ] No data leakage              │
│   [ ] Functions <50 lines         [ ] Seeds are explicit           │
│   [ ] Comments explain why        [ ] NaN handling defined         │
│   [ ] No deep nesting             [ ] Metadata is logged           │
│                                                                     │
│   TESTING                         PRODUCTION                        │
│   ───────                         ──────────                        │
│   [ ] Happy path tested           [ ] Errors are clear             │
│   [ ] Edge cases covered          [ ] Logging in place             │
│   [ ] Tests are deterministic     [ ] No secrets in code           │
│   [ ] Tests run fast              [ ] Input validation             │
│                                                                     │
│   THE GUT CHECK                                                     │
│   ─────────────                                                     │
│   [ ] Would I debug this at 2am?                                   │
│   [ ] Am I proud of this code?                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

*"Code review is not about finding bugs. It's about raising the bar for what 'good enough' means."*
