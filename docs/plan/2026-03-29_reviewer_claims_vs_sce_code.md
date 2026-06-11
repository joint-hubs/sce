# Reviewer Claims vs `sce` Code

## Date
2026-03-29

## Purpose

This note compares the main ICML reviewer claims against what is actually implemented in the `sce` repository.

The goal is to separate:
- claims the code supports well
- claims that are only partially supported
- claims where reviewers are right
- hidden risks in the current experimental protocol

---

## Executive Summary

After reviewing `sce`, several reviewer criticisms should be **softened**, because the implementation is more advanced than the paper currently communicates:
- manual ordered hierarchy specification is **not** required in the default workflow
- the engine auto-detects categorical grouping variables
- interactions are built automatically
- there is optional cleanup for redundancy and leakage-like features
- there is a nontrivial feature/model search stack

However, several reviewer concerns remain **correct**, and one additional issue is potentially more serious than the current reviews explicitly state:
- the paper still lacks the strongest relevant baselines
- the model-agnostic claim is only weakly supported in current experiments
- broad generalization to time series / cross-domain tasks is not established
- some selection/search layers are experimental, not production-grade
- the experiment runner appears to enrich the full dataset before train/test split, which may create an evaluation protocol problem

---

## Claim-by-Claim Assessment

## 1. "The method relies heavily on manual hierarchy specification and domain knowledge"

### Verdict
**Partially incorrect as stated. Should be rebutted, but only carefully.**

### What the code shows
- `ContextConfig.hierarchy` is deprecated
- default mode uses `categorical_cols=None`, which triggers auto-detection
- grouping columns are auto-detected using dtype + cardinality rules
- interactions are generated automatically across detected categorical columns
- all categorical columns are treated as peers, not as a strict user-authored ordered tree

### Relevant code evidence
- `sce/config.py`: `detect_categorical_columns()` and `ContextConfig.get_categorical_cols()`
- `sce/config.py`: `hierarchy` explicitly marked deprecated
- `sce/stats.py`: `compute_aggregations()` creates single-column and interaction groupings automatically
- `sce/engine.py`: fit path resolves categorical columns automatically

### What remains true
- the method still assumes that meaningful grouping structure exists in the data
- this is a structural assumption, not the same as saying the user must hand-engineer a full hierarchy

### Rebuttal position
Say:
"The current implementation does not require manually specified ordered hierarchies in the default workflow. SCE can auto-detect grouping variables and construct grouping levels and interactions automatically. The real assumption is the presence of meaningful categorical/group structure in the data."

---

## 2. "Novelty is limited; this is close to target encoding / grouped target statistics"

### Verdict
**Reviewers are basically right. Concede in part.**

### What the code shows
- core engine is a structured system of grouped target-derived statistics
- those statistics are computed across many groupings and interactions
- leakage prevention and fold-variance add engineering value

### What the code does not show
- no fundamentally different primitive from target-statistic-based feature construction
- no built-in benchmark suite against canonical target encoding baselines

### Relevant code evidence
- `sce/stats.py`: grouped aggregation over target variable is central mechanism
- `sce/engine.py`: cross-fitting and fold aggregation machinery
- no CatBoost / target encoding comparator in `scripts/run.py`

### Rebuttal position
Concede primitive-level novelty is incremental.
Defend system-level novelty:
- hierarchical multi-level representation
- leakage-safe construction
- fold-variance uncertainty features
- explicit context systems instead of single mean encoding

---

## 3. "There are not enough strong relevant baselines"

### Verdict
**Correct. This is one of the strongest reviewer points.**

### What the code shows
- baseline in experiments is mainly XGBoost on base features
- the code supports many SCE subset strategies and XGBoost presets
- the code does not implement target encoding, CatBoost ordered statistics, TabNet, TabTransformer, or FT-Transformer baselines

### Relevant code evidence
- `scripts/run.py`: experiments revolve around base vs SCE-enriched feature sets
- `sce/search.py`: strategies are `baseline`, `context_only`, `base_context`, significance-selected subsets, ablations
- `sce/search.py`: `train_model()` only supports XGBoost in search mode

### What this means
The selection/search stack is sophisticated, but it is searching **within the SCE family** plus one base baseline, not across the strongest neighboring method families.

### Rebuttal position
Do not fight this point.
Agree and add the missing baselines if at all possible:
- standard target encoding
- mean/count hierarchical encoding baselines
- CatBoost baseline
- optionally one deep tabular baseline if time permits

---

## 4. "The model-agnostic claim is not fully validated"

### Verdict
**Correct in the paper-evidence sense, even if the library API is generic.**

### What the code shows
- the transformer itself is scikit-learn compatible and can theoretically feed any downstream model
- `create_sce_pipeline()` is generic
- actual search / reporting experiments are centered on XGBoost presets
- search code explicitly supports only XGBoost

### Relevant code evidence
- `sce/pipeline.py`: generic sklearn pipeline support
- `sce/search.py`: `train_model()` raises unless model type is XGBoost
- `scripts/run.py`: experiment flow optimized around XGBoost presets and XGBoost-focused reports

### Rebuttal position
Say:
"The representation layer is model-agnostic at the API level, but we agree that the current empirical validation is concentrated on XGBoost-style learners. We will narrow the empirical claim accordingly."

---

## 5. "Feature dimensionality / redundancy / regularization are underexplained"

### Verdict
**Partially correct. The code has answers, but the paper is not surfacing them clearly enough.**

### What the code shows
- cleanup pipeline can remove constant, leakage-like, correlated, high-VIF, and hierarchy-redundant features
- search layer supports LM significance selection, tree-importance selection, ablation, and pruning
- model presets define different XGBoost regularization profiles indirectly via depth / estimators / learning rate / subsampling

### Relevant code evidence
- `sce/cleanup.py`: cleanup pipeline
- `sce/selection.py`: LM p-value feature selection
- `sce/search.py`: significance selection, ablations, iterative pruning support
- `sce/model_presets.py`: model preset families
- `scripts/run.py`: saves LM statistics, pruning traces, aggregated feature importance

### Important caveat
- `selection.py` and `search.py` are marked experimental with low test coverage
- this makes them useful as supporting tools, but weaker as the foundation of a central scientific claim

### Rebuttal position
You can explain more here than the current paper does.
But do not oversell the experimental selection stack as fully mature.

---

## 6. "Backoff and sparse-group robustness are not evaluated"

### Verdict
**Mostly correct. The mechanism exists, but the evidence is thin unless reports were actually surfaced.**

### What the code shows
- backoff exists
- backoff depth features can be added
- runner logs backoff statistics when backoff depth columns are present

### Relevant code evidence
- `sce/engine.py`: applies hierarchical backoff
- `sce/stats.py`: aggregation respects `min_group_size`
- `scripts/run.py`: logs mean backoff depth, max depth, percent of backoff usage if depth features exist

### What is missing
- this does not appear to be a central reported result in the current paper
- existence of code is not the same as evidence in the paper

### Rebuttal position
Say the mechanism exists and can be summarized empirically, but do not claim the current paper already characterizes it fully unless you actually add the diagnostics.

---

## 7. "There is no seed variance / confidence interval / uncertainty reporting"

### Verdict
**Mostly correct for the paper. Partially softened by code support, but not solved.**

### What the code shows
- random seeds are fixed in many places (`random_state=42`)
- fold-variance features exist as part of representation uncertainty
- but this is not the same as reporting experimental uncertainty across reruns / splits

### Relevant code evidence
- `sce/engine.py`: KFold uses fixed random_state
- `sce/search.py`: model search and RF selectors use fixed random_state
- `scripts/run.py`: train/test split sampling also uses fixed random_state

### What is missing
- repeated seeds
- repeated train/test splits
- bootstrap CIs for headline metrics

### Rebuttal position
Concede this point.
Having fixed seeds supports reproducibility, but not robustness estimation.

---

## 8. "No deep tabular comparison"

### Verdict
**Correct.**

### What the code shows
- no TabNet / TabTransformer / FT-Transformer implementation or runner support found

### Rebuttal position
Do not resist this criticism.
Either add one baseline or narrow the claim to interpretable statistical context for boosted-tree settings.

---

## 9. "No time-series or classification validation"

### Verdict
**Correct.**

### What the code shows
- repo language sometimes mentions time series
- experiment configs and packaged datasets are regression datasets in real estate
- no clear classification benchmark pipeline found
- no clear time-series benchmark pipeline found

### Relevant code evidence
- dataset configs in `configs/` are all real-estate regression tasks
- metrics in `scripts/run.py` are RMSE and R² focused

### Rebuttal position
Narrow the scope. Do not defend broad time-series generality from the current evidence.

---

## 10. "The strongest gains may be target leakage through fine groups"

### Verdict
**This deserves careful treatment. The code has both a defense and a risk.**

### What the code supports in your favor
- cross-fitting is implemented in the engine
- min group size exists
- fold-variance features and cleanup exist
- leakage-like features can be filtered in cleanup

### Relevant code evidence
- `sce/engine.py`: cross-fitted feature generation
- `sce/config.py`: `min_group_size`, fold variance, cleanup config
- `sce/cleanup.py`: suspiciously high target-correlation removal

### But there is an important protocol risk
The experiment runners appear to call `engine.fit_transform(df)` on the full dataset **before** the final train/test split.

This means:
- each row gets out-of-fold features relative to folds of the full dataset
- if the dataset is split into train/test **after** this step, then test-row context features may depend on target information from other test rows
- that is not the same as fitting on train and transforming test using train-only statistics

### Relevant code evidence
- `scripts/run.py`: in both `run_search_experiment()` and `run_experiment()`, `engine.fit_transform(df)` is applied before train/test split

### Why this matters
Even with cross-fitting, applying it on the full dataset before holdout splitting can blur the clean separation expected in a predictive evaluation.

This does **not** automatically prove the results are invalid, but it creates a real review risk and should be checked immediately.

### Recommendation
Treat this as a priority audit item.
At minimum:
- rerun one dataset with the clean protocol: split first, fit SCE on train, transform test using train-derived stats only
- compare headline deltas

If results hold, this strengthens the rebuttal a lot.
If they shrink, better to learn that now than after escalation.

---

## 11. "The method depends strongly on XGBoost / nonlinear learners"

### Verdict
**Likely correct empirically, not fully settled by code.**

### What the code shows
- the search machinery is built for XGBoost
- there is no strong evidence in current experiment code for broad learner comparisons

### Rebuttal position
Admit the current empirical center of gravity is XGBoost.
Do not present current results as proving equal utility across learner classes.

---

## 12. "The explanation of why context-only can underperform while context dominates importance is unclear"

### Verdict
**This can be explained well from the method and code.**

### Best explanation
Context features and base features are not substitutes.

Interpretation:
- context features define the local distribution and neighborhood priors
- base features identify the individual instance within that local distribution
- context-only can miss instance-level positioning
- base-only can miss neighborhood normalization
- combined models win because both roles are needed

This is fully consistent with a feature-importance pattern where contextual features dominate splits, while base features remain necessary for final discrimination.

---

## Overall Rebuttal Strategy After Code Review

## Strongly defend
- auto-detection of grouping structure
- leakage-safe engineering intent in the core engine
- nontrivial redundancy cleanup and feature pruning support
- contextual-system framing instead of one hand-engineered hierarchy

## Partially defend, partially narrow
- model-agnostic claim
- hierarchy/domain-knowledge discussion
- dimensionality handling
- backoff mechanism

## Concede
- too few strong external baselines
- no deep tabular benchmark
- no time-series / classification validation
- too-broad empirical scope
- insufficient uncertainty reporting across seeds/splits

## Audit immediately
- whether `fit_transform(df)` before train/test split contaminated holdout evaluation

---

## Most Important Next Actions

1. Audit the evaluation protocol in `scripts/run.py` and rerun at least one dataset with split-first evaluation.
2. Add the missing closest baselines: target encoding, simple hierarchical mean/count baselines, CatBoost if feasible.
3. Export backoff-depth and effective-group-size diagnostics.
4. Add repeated split / seed variance tables.
5. Narrow the paper claims to match what is actually demonstrated.