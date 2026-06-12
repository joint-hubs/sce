# Research Roadmap: SCE → Shapley-Bayesian Core → Self-Optimizing Data Layer → SGN

> Status: PROPOSAL (2026-06-12). Synthesizes: `timeseries-context-engineering/docs/01–13`,
> prof. SMS discussion (2026-05-18), ICML 2026 reviews, and the current state of
> `stat-context` (0.4.0, 5 report-grade datasets, leakage diagnostics suite).
> Owner: Mateusz Stachowicz.

---

## 1. The core thesis (what we are actually building)

Mateusz's stated vision: *given a model and a prediction target, a layer that
automatically engineers and optimizes the feature set for that specific use
case — so the data scientist receives a tailored, ready dataset instead of
doing manual feature work. "A model before the real model."*

Reframed in research terms, this is **model-aware, leakage-safe automated
context engineering**, built from three components that already exist in
draft form:

1. **SCE (exists, working)** — a constrained *grammar* of candidate features:
   categorical groups × interactions × statistical operators × hierarchical
   backoff. This is what makes the search space tractable and auditable,
   unlike open-ended AutoFE.
2. **Bayesian stabilization (drafted)** — James-Stein / hierarchical shrinkage
   makes every candidate feature a *reliable player* (small groups don't
   produce garbage estimates; posterior variance quantifies trust).
3. **Shapley valuation (drafted)** — cooperative game theory assigns each
   candidate context its *marginal value for the given downstream model*,
   handling redundancy between overlapping groups (city vs zip-code) in a
   principled, axiomatic way.

The loop: **generate (SCE grammar) → stabilize (Bayes) → value (Shapley w.r.t.
the target model) → prune/select → emit optimized dataset + attribution
report.** That loop *is* the "data self-optimization" — and each iteration is
verifiable with the existing leakage diagnostics gate.

## 2. Critical assessment — does this make sense?

**Yes, with three conditions.**

### 2.1 The professor is right about positioning

SCE-as-architecture has no breakout strength (reviewers said it; the
benchmarks confirm modest 1–11% gains). But SCE-as-*search-space* plus
**Shapley-Bayesian valuation as the central contribution** is a coherent new
current: *feature engineering as a cooperative game with Bayesian-stabilized
players*. The Bayes–game-theory component must be the headline, SCE the
substrate, SGN the eventual architecture. This matches the prof's "nowy prąd
połączenia bayesa i TG z feature engineeringiem — prędzej".

### 2.2 The competitive landscape (must position against, not ignore)

| Prior work | What it does | Our differentiation |
|---|---|---|
| **Data Shapley** (Ghorbani & Zou, ICML 2019) | Shapley value of *data points* | We value *features/contexts*; Bayesian value function is novel |
| **OpenFE** (ICML 2023) | Automated feature generation + 2-stage pruning | Greedy/incremental, leakage-prone, no uncertainty; we offer axiomatic credit + posterior variance + leakage proofs |
| **featuretools / DFS** | Relational aggregation primitives | No target-aware valuation, no leakage protocol |
| **CAAFE / LLM-based AutoFE** | LLM proposes features | Non-deterministic, unauditable; we are deterministic + attributable |
| **SHAP** (Lundberg & Lee) | Post-hoc explanation of a trained model | We use Shapley *upstream* — to decide what enters the model |
| **CatBoost ordered TS** | Leakage-safe target statistics | Single statistic, no group valuation, no redundancy handling |

The unique claim no competitor makes: **the emitted feature set comes with an
axiomatic attribution certificate (φ per context) and a falsifiable
no-leakage certificate (permuted-target / shuffled-groups diagnostics).**
That combination — *valuation + verification* — is the moat. Our diagnostics
suite, built for the SCE paper remediation, becomes a first-class scientific
instrument here, not just hygiene.

### 2.3 Honest risks

| Risk | Severity | Mitigation |
|---|---|---|
| Shapley cost explodes (2^(p+m) coalitions) | HIGH | SCE grammar bounds m; Monte-Carlo permutation sampling (already drafted in Part 11); hierarchy-aware pruning (children share parent's coalition stats) |
| "Novelty = SHAP + target encoding" reviewer attack | MEDIUM | Theorems 5.2–5.4 (efficiency/redundancy/null-context) + the Bayesian value function v(S) are the formal novelty; lead with them |
| Gains over XGBoost-with-SCE may be small | MEDIUM | Primary claims: *automation* (hours of DS work → minutes) and *auditability*, with accuracy parity; accuracy gains are secondary |
| "Self-optimization" oversell | LOW | Never claim universal AutoML; scope = tabular + hierarchical categorical structure (where we have report-grade evidence) |
| SGN hypothetical numbers in docs (Part 12 "Results (Hypothetical)") | HIGH if leaked | Mark clearly; never cite until real runs pass the same diagnostics gate as SCE |

### 2.4 What I would NOT do

- **Don't merge everything into one paper.** Prof is right: SCE→SGN in one
  paper is too much material. Three papers (below), each self-contained.
- **Don't start with SGN.** It depends on the valuation core being credible,
  and deep-tabular baselines (TabNet/FT-Transformer) are brutal to beat
  honestly. SGN inherits credibility from papers 1–2.
- **Don't build the agentic/LLM wrapper first.** The "smart agent" docs are
  infrastructure, not science. They wrap the loop after the loop exists.

## 3. The three-paper program

### Paper 1 — SCE (foundation; mostly done)
*"Statistical Context Engineering: leakage-safe hierarchical target
statistics"* — the current ICML rebuttal line. Value: establishes the
substrate + the diagnostics protocol. **Must exist first** ("SCE musi istnieć
wcześniej, żeby SGN miało z czego czerpać"). Status: 5/8 datasets
report-grade, library 0.4.0 ready, paper figures need regeneration from the
post-remediation runs.

### Paper 2 — Shapley-Bayesian Feature Valuation (the new current; CENTRAL)
*"Cooperative games for feature valuation: a Shapley-Bayesian framework for
automated context engineering."* Contributions:
1. Value function from Bayesian posteriors (v(S) = quality of precision-
   weighted coalition prediction) — Theorem 5.1.
2. Properties: efficiency, redundancy-sharing, null-context ⇒ automatic
   feature selection (Theorems 5.2–5.4 + Corollary 5.1).
3. The generate–stabilize–value–prune loop as an algorithm, with leakage
   certificates.
4. Benchmarks vs OpenFE / featuretools / greedy forward selection on the
   5 report-grade datasets + 2–3 public AutoFE benchmarks.
Target: ICML/NeurIPS (theory+method). This is where the impact concentrates:
it names the new current.

### Paper 3 — SGN (architecture; later)
*"Shapley-Gated Networks"* — differentiable Bayesian encoder + Shapley gate
with axiomatic regularization (Part 12). Only after Paper 2's valuation is
published, and only with real (not hypothetical) benchmark numbers that pass
the diagnostics gate. Target: NeurIPS/ICLR.

## 4. Engineering roadmap (what gets built, in order)

### Phase 0 — Close SCE (now → end of June 2026)
- [ ] Push + CI green; release `stat-context` 0.4.0 to PyPI.
- [ ] UAE full-data diagnostics (sales_uae permuted +24.5% must be explained);
      re-promote or permanently drop experimental configs.
- [ ] Regenerate paper figures/tables from report-grade runs only.
- [ ] Finish ICML rebuttal cycle for Paper 1.

### Phase 1 — Shapley-Bayesian core in `stat-context` (Jul–Sep 2026)
New subpackage `sce.valuation` (working name), built ON the existing engine:
- [ ] `BayesianGroupEstimator`: James-Stein/hierarchical shrinkage with
      posterior variance per group (replaces/upgrades current backoff; the
      "ustabilizować bayesa" thread).
- [ ] `CoalitionGame`: value function v(S) over context groups, evaluated on
      the *internal validation split* (reuse the search-fix protocol — test
      touched once, selection never sees it).
- [ ] `ShapleyValuator`: exact (small m) + Monte-Carlo permutation sampling
      (large m), with bootstrap CIs.
- [ ] Extended coalition (raw features + contexts as players) — Part 11.
- [ ] Experiments E1–E8 from Part 13 on the 5 report-grade datasets.
- [ ] **Gate**: every valuation experiment runs under the same diagnostics
      suite; permuted-target failure = result not citable. Non-negotiable.

### Phase 2 — Self-optimizing data layer (Oct–Dec 2026)
The product form of Mateusz's vision:
- [ ] `sce.optimize(model, df, target, time_col=None) -> OptimizedDataset`:
      runs the full loop against the *user's* model class (model-aware: the
      value function trains the actual downstream model on coalitions).
- [ ] Output artifact: enriched parquet + `attribution.json` (φ per context,
      posterior variances, backoff depths) + `certificate.json` (diagnostics
      results, git SHA, config hash) — the audit trail is part of the API.
- [ ] DataOps layer ("automatyzacja przygotowania danych z wielu źródeł"):
      declarative source manifests → deterministic parquet, extending the
      existing configs/manifest/checksum machinery; pre/post-processing as
      recorded, replayable steps.
- [ ] MLOps: model registry of supported downstream models (xgboost core;
      lightgbm/catboost via extras; sklearn estimators generically).
- [ ] Benchmark harness vs OpenFE/featuretools/CAAFE → experimental section
      of Paper 2; applied/systems paper (KDD) optional spin-off.

### Phase 3 — SGN prototype (2027, after Paper 2 submission)
- [ ] PyTorch implementation per Part 12 (Differentiable Bayesian Encoder,
      Shapley Gate with efficiency/null/symmetry regularizers).
- [ ] Ablations: gate vs plain attention; learned vs fixed shrinkage —
      isolating whether the game-theoretic inductive bias earns its cost.
- [ ] Honest baselines: tuned XGBoost+SCE, FT-Transformer, TabNet; temporal
      variants on Rossmann/Walmart/M5 (Temporal Shapley Gate, Part 12.7.3).

## 5. Where the work lives

- **`sce` repo (`stat-context`)**: stays the production library — engine,
  diagnostics, valuation (Phase 1), optimize API (Phase 2). One library, one
  protocol, one audit trail.
- **`timeseries-context-engineering` repo**: research notebook / theory docs /
  SGN prototyping. Graduates code into `stat-context` only after it passes
  the diagnostics gate. Hypothetical numbers stay quarantined here.

## 6. Success criteria (1 year)

| Milestone | Evidence |
|---|---|
| Paper 1 accepted/arXiv'd | citable SCE foundation |
| `sce.valuation` merged | Shapley CIs reproducible on 5 datasets, all diagnostics green |
| Paper 2 submitted (ICML/NeurIPS 2027 cycle) | with leakage certificates as a named contribution |
| `sce.optimize` MVP | one command: raw df → optimized dataset + attribution, ≥ parity with hand-tuned SCE configs |
| SGN go/no-go decision | real ablation numbers, not Part-12 hypotheticals |
