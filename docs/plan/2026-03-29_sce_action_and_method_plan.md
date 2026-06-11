# SCE Action Plan and Method Expansion Plan

## Date
2026-03-29

## Goal

This note has two purposes:
1. define the highest-ROI actions for the current ICML rebuttal window
2. define how SCE should evolve as a method so the next version is scientifically fuller, not just better packaged

---

# Part I. Immediate Action Plan for ICML

## Summary

The current situation is not just "add a few baselines".

There are three separate layers to fix:
- **evaluation correctness**
- **comparison fairness**
- **claim calibration**

The order matters.

If the evaluation protocol is not clean, no amount of rebuttal wording will save the paper.

---

## Priority 0. Audit the Evaluation Protocol

### Why

The current experiment runner in `sce/scripts/run.py` appears to do `engine.fit_transform(df)` on the full dataset before the final train/test split.

Even with cross-fitting, that is not equivalent to the clean predictive protocol reviewers expect.

### Risk

Potential issue:
- test-row context features may be informed by target structure from other test rows
- this weakens the holdout interpretation

### Required action

Create a clean protocol branch of the experiment pipeline:

#### Protocol A: current legacy evaluation
- enrich full dataset first
- split later

#### Protocol B: clean predictive evaluation
- split first
- fit SCE only on training data
- transform test set using train-derived statistics only
- train downstream model on train enriched features
- evaluate on test enriched features

### Success criterion

For at least 1-2 datasets:
- compare Protocol A vs Protocol B
- if performance remains close, this becomes strong rebuttal evidence
- if performance drops materially, update the paper immediately and stop using the legacy result as headline evidence

### Engineering tasks

1. Add a clean experiment path in `scripts/run.py`
2. Add a regression test that enforces split-first evaluation
3. Save both protocols side-by-side for one dataset to a CSV report

---

## Priority 1. Add the Missing Closest Baselines

### Why

This is the strongest reviewer criticism, and they are right.

The current search stack is rich **inside SCE**, but the scientific comparison set is still too narrow.

### Minimum baseline pack

Add these first:

#### Baseline A. Base model only
- XGBoost with raw base features

#### Baseline B. Plain target mean encoding
- one statistic per categorical grouping
- leakage-safe / out-of-fold version

#### Baseline C. Hierarchical mean + count
- grouped mean + count only
- same group generation logic as SCE
- no extra quantiles / std / fold variance

#### Baseline D. Hierarchical mean + std + count
- a stronger but still simple statistical baseline

#### Baseline E. CatBoost
- if feasible with the same raw feature set
- especially important because reviewers cited ordered statistics

### Optional second wave

Only if time remains:
- LightGBM categorical baseline
- TabNet or FT-Transformer

### Key experimental question

The right scientific question is not:
"Does SCE beat a weak baseline?"

It is:
"Does a full multi-level statistical context system beat simpler leakage-safe target-statistic baselines built on the same grouping structure?"

That is the most defensible comparison.

---

## Priority 2. Add Stability and Uncertainty Reporting

### Why

Fixed `random_state=42` helps reproducibility, but not robustness.

### Required outputs

For at least the strongest two datasets:
- repeated train/test splits or repeated CV
- mean RMSE and std RMSE
- mean R² and std R²
- optionally bootstrap confidence intervals for delta vs baseline

### Minimum acceptable table

| Dataset | Method | Mean RMSE | Std RMSE | Mean ΔRMSE | Mean R² |
|---------|--------|-----------|----------|------------|---------|

### Why this matters scientifically

This directly answers:
- "are gains stable?"
- "are improvements just split luck?"

---

## Priority 3. Export Backoff and Effective Group Diagnostics

### Why

Reviewers are suspicious that gains may come from near-identifiable tiny groups.

The code already contains enough machinery to report this, but the paper does not show it.

### Required diagnostics

For each dataset, export:
- percentage of rows using native group statistics with no backoff
- percentage of rows that needed backoff
- mean backoff depth
- max backoff depth
- distribution of effective group sizes
- fraction of groups removed by `min_group_size`

### Optional helpful diagnostic

Performance by backoff regime:
- low backoff subset
- medium backoff subset
- high backoff subset

This gives a much cleaner response to the leakage/sparsity objection.

---

## Priority 4. Clarify the Feature Selection Story

### Why

Right now the repo has three different stories mixed together:
- core context engine
- cleanup pipeline
- experimental search / selection

Reviewers may read this as hand-wavy or over-engineered unless the roles are separated clearly.

### Correct story

#### Core method
- automatic grouping discovery
- grouped statistical context construction
- cross-fitting
- fold variance

#### Optional cleanup
- remove degenerate, leakage-like, correlated, and hierarchy-redundant features

#### Experimental optimization layer
- LM significance selection
- tree-importance selection
- combinatorial search
- ablations and iterative pruning

### Required paper change

Make clear that the **scientific claim** is about the core representation, not about the entire experimental search stack.

---

## Priority 5. Narrow the Paper Claims

### What to narrow immediately

Replace or soften:
- "tabular and time-series" -> "hierarchical tabular regression" unless new evidence is added
- "model-agnostic" -> "representation is model-agnostic at the API level; current empirical study focuses on boosted-tree learners"
- strong theory language -> explanatory interpretation under structured heterogeneity

### What to keep

Keep and strengthen:
- explicit contextual statistical representation
- leakage-safe construction intent
- interpretability and auditability
- practical value in structured domains

---

## Suggested 5-Day Execution Order

### Day 1
- audit evaluation protocol
- implement split-first clean run
- rerun 1 dataset

### Day 2
- add simple target encoding / grouped mean-count baselines
- rerun on 1-2 datasets

### Day 3
- export backoff/group-size diagnostics
- export repeated split variance table

### Day 4
- update paper framing and limitations
- draft rebuttal using audited results only

### Day 5
- optionally add CatBoost or one deep baseline
- polish response tables and figures

---

# Part II. How to Make SCE a Fuller Method

## Summary

Right now SCE is strongest as a **practical representation layer**.

To become a fuller method in the research sense, it should grow in five directions:
- cleaner statistical foundation
- stronger context discovery
- reliability-aware aggregation
- broader task coverage
- stronger evaluation design

---

## Direction 1. Separate "Context Discovery" from "Context Summarization"

### Current state

The engine mostly assumes grouping variables exist and then summarizes targets within those groups.

### Limitation

That still leaves the hardest scientific question underdeveloped:
how should the system discover which groupings are worth using?

### Full-method upgrade

Define SCE as two-stage:

#### Stage A. Context discovery
- detect categorical columns
- score candidate grouping levels
- score interactions
- select a compact context basis

#### Stage B. Context summarization
- compute stable multi-statistic summaries on selected groups
- attach reliability signals

### Good next-step mechanisms

- candidate scoring by out-of-fold gain
- mutual information or residual reduction per grouping
- coverage × stability × gain scoring
- redundancy-aware selection of grouping levels

### Why this matters

It moves SCE from "feature engineering system" toward a more complete representation-learning method for structured tabular data.

---

## Direction 2. Make Reliability a First-Class Object

### Current state

You already have pieces of this:
- fold variance
- min group size
- backoff depth
- cleanup by suspicious target correlation

### Full-method upgrade

Define a formal per-context reliability score.

Potential components:
- group size
- fold variance
- backoff depth
- coverage across folds
- shrinkage intensity

### Example

For each contextual feature block, define:

$$
R_g = f(n_g, \sigma_{fold,g}, d_{backoff,g}, c_g)
$$

Then either:
- expose `R_g` as a feature
- use it to weight context blocks
- use it to regularize or prune unstable contexts

### Why this matters

This gives a much cleaner answer to reviewers asking when the method should be trusted.

---

## Direction 3. Add Shrinkage-Based Context Estimation

### Current state

Current summaries are mainly empirical grouped statistics plus backoff.

### Full-method upgrade

Move from raw grouped estimators toward shrinkage estimators:
- empirical Bayes shrinkage toward coarser levels or global prior
- hierarchical smoothing for small groups
- robust estimators for heavy-tailed targets

### Concrete variants

#### Variant A. Global shrinkage
- group mean shrunk toward global mean by group size

#### Variant B. Hierarchical shrinkage
- child group shrunk toward parent group
- parent group shrunk toward global mean

#### Variant C. Robust shrinkage
- median-centered shrinkage for heavy-tailed price distributions

### Why this matters

This would make the method statistically stronger than plain target aggregation and would directly improve the novelty story.

---

## Direction 4. Add Redundancy-Aware Context Selection at the Group Level

### Current state

You prune feature redundancy after generation, but there is still no clean group-level selection principle.

### Full-method upgrade

Select context groups, not only features.

Instead of generating everything and cleaning later:
- generate candidate groups
- estimate marginal out-of-fold gain for each group
- remove redundant groups whose contribution is subsumed by others

### Possible selection criteria

- incremental gain in validation RMSE
- information gain conditional on already selected groups
- diversity penalty across overlapping group definitions

### Why this matters

This makes the method more principled, more compact, and less vulnerable to the "feature explosion" criticism.

---

## Direction 5. Support Additional Tasks Properly

### Current state

The public story says tabular and time-series, but the packaged evaluation is still regression-heavy and real-estate-specific.

### Full-method upgrade

Add task-specific SCE variants:

#### Classification SCE
- grouped class probabilities
- grouped class entropy
- grouped log-odds and reliability

#### Time-Series SCE
- lag-aware grouped context
- rolling group summaries
- leakage-safe temporal cross-fitting
- regime-aware context by segment/time window

#### Ranking / retrieval style variant
- relative context scores for ordering within groups

### Why this matters

This is the cleanest path to restoring the broader method claim in a future paper.

---

## Direction 6. Add Causal and Distribution-Shift Robustness

### Current state

The method assumes group structure is informative and reasonably stable.

### Full-method upgrade

Study and harden against:
- unseen groups
- group shift
- distribution drift within known groups
- weak or noisy grouping variables

### Concrete upgrades

- explicit unknown-group fallback regime
- calibration under group shift
- sensitivity analysis for misspecified grouping variables
- adversarial or stress-test evaluation under perturbed group definitions

### Why this matters

This directly answers practical deployment concerns and would strengthen significance.

---

## Direction 7. Define a Cleaner Theory Target

### Current state

The theory currently risks sounding broader than the evidence.

### Full-method upgrade

The theory should target a narrower but more defensible question:

"When does a leakage-safe contextual summary of local conditional target structure improve downstream learning over the raw covariate representation alone?"

### Better theory components

- excess-risk decomposition under structured heterogeneity
- bias-variance trade-off with grouped shrinkage
- reliability-weighted context aggregation
- sufficient conditions under which group context helps
- failure modes when group structure is weak, sparse, or unstable

### Why this matters

This gives the paper a clearer scientific center instead of a broad intuition statement.

---

## Direction 8. Build a Standardized Benchmark Suite

### Current state

The current benchmark is useful but narrow.

### Full-method upgrade

Create a benchmark matrix with:
- structured tabular regression
- structured tabular classification
- time-series with grouped regimes
- synthetic benchmarks with controllable hierarchy strength

### Benchmark dimensions to vary

- group cardinality
- group sparsity
- hierarchy quality
- overlap/redundancy of contexts
- amount of shift between train and test

### Why this matters

This would turn SCE into a reproducible research program instead of a strong single-domain case study.

---

## A Fuller Future Definition of SCE

If SCE matures well, the method should be described like this:

"SCE is a structured representation-learning framework for datasets with latent or explicit grouping structure. It automatically discovers candidate grouping contexts, estimates leakage-safe contextual statistics with reliability-aware shrinkage, selects a compact non-redundant context basis, and exposes the resulting representation to downstream learners across regression, classification, and temporal prediction settings."

That is a much fuller method than the current version, and it would be far harder to dismiss as only a practical encoding trick.

---

## Recommended Development Roadmap

## Phase 1. Clean the science
- fix evaluation protocol
- add closest baselines
- add uncertainty reporting
- narrow claims

## Phase 2. Strengthen the method
- formal context discovery
- reliability scoring
- shrinkage estimators
- group-level redundancy selection

## Phase 3. Broaden the scope honestly
- classification variant
- time-series variant
- shift robustness
- standardized benchmark suite

## Phase 4. Write the next paper around the stronger core
- one clear method definition
- one clean theory target
- broader evidence across tasks and domains

---

## Bottom Line

For the current ICML cycle, the best move is to make SCE look **more honest and more rigorous**.

For the next cycle, the best move is to make SCE look **less like a clever statistical encoding package and more like a full structured context representation method with explicit discovery, reliability, and generalization machinery**.