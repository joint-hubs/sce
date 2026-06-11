# ICML Rebuttal Plan for SCE

## Date
2026-03-29

## Situation Summary

The review pattern is consistent.

This submission is not being rejected because reviewers think the empirical signal is fake.
It is being rejected because the paper currently overclaims breadth and novelty relative to what is shown.

The core message from the panel is:
- the method looks practically useful
- the gains look real
- the presentation is readable enough for some reviewers
- but the paper currently reads broader and more fundamental than the evidence supports

In short: the biggest gap is positioning, scope control, and missing comparison baselines.

## Review Landscape

| Reviewer | Score | Main Theme |
|----------|-------|------------|
| aRsy | 2 | blind violation, novelty limited, scope too narrow, hierarchy hand-specification |
| RzPC | 3 | strongest review; asks for target-encoding baselines, theory clarification, uncertainty, robustness, broader scope |
| jNzT | 2 | practical value acknowledged, but novelty incremental, no deep baselines, no cross-domain validation |
| Tjq1 | 2 | paper logic and mechanism not explained tightly enough |

Net assessment:
- Current probability of rescue is low if the response is only defensive.
- There is still a plausible rebuttal if the response is sharp, honest, and evidence-backed.
- The highest-ROI path is to narrow the contribution and answer the strongest missing baseline questions directly.

## Non-Negotiable Reframing

These changes should shape the rebuttal, and also the next revision of the paper.

## Important Correction After Inspecting `sce`

After reviewing the actual implementation repo, one point needs to be corrected relative to the initial reading of the paper and reviews.

The production SCE engine does **not** currently require a manually specified ordered hierarchy in the narrow sense of hand-authoring a chain like `country -> region -> city -> neighborhood`.

What the code actually does:
- `ContextConfig.hierarchy` is explicitly marked **deprecated**
- categorical grouping columns can be **auto-detected** from dtypes and cardinality
- all categorical columns are treated as peers rather than as a strict ordered tree
- the engine automatically builds single-column and interaction-level groupings
- cross-fitting and fold-variance features are part of the core engine
- optional cleanup removes leakage-like features, correlated features, and hierarchy-redundant features

What still remains true:
- the method still depends on the existence of **meaningful categorical/group structure** in the data
- this is weaker than "hand-specified domain hierarchy is required", but stronger than "no domain structure matters"

This means the rebuttal should avoid conceding too much on the "manual hierarchy engineering" point.
The correct framing is:

"The current implementation does not require the user to manually specify an ordered domain hierarchy. In the default mode, SCE auto-detects suitable categorical grouping variables, constructs grouping levels and interactions automatically, and applies leakage-safe aggregation over those groups. What remains necessary is that the dataset contain semantically meaningful grouping structure; this is a structural assumption of the method, not manual feature crafting by the user."

## Practical Architecture Summary From `sce`

The implementation has three distinct layers that should be described separately in rebuttal or revision:

### 1. Core engine: production-ready

This is the strongest part of the system.

Core components:
- `engine.py`
- `config.py`
- `stats.py`

Main capabilities:
- auto-detection of categorical grouping columns
- cross-fitted aggregation
- interaction generation across categorical columns
- fold-variance uncertainty features
- optional hierarchical backoff and backoff-depth features

This is the part that supports the main methodological claim.

### 2. Cleanup / pruning layer: optional but important

This is more than a toy utility and matters for the rebuttal because reviewers asked about dimensionality, redundancy, and leakage-like artifacts.

Capabilities:
- constant-feature removal
- suspicious target-correlation filtering
- pairwise correlation pruning
- optional VIF pruning
- hierarchy-redundancy removal

This helps answer reviewer concerns about very wide feature spaces and brittle redundant context features.

### 3. Search / selection layer: sophisticated but experimental

This is where the story needs precision.

The repo contains a genuinely complex search-and-selection layer:
- random search over context subsets
- multiple XGBoost presets
- significance-based selection via linear-model p-values
- tree-importance selection
- ablation by removing best / worst features
- iterative pruning and aggregated importance analysis

However, this layer is explicitly labeled in code as **experimental** with low test coverage.

So the honest position is:
- yes, the implementation includes nontrivial automatic feature/model selection
- but the core scientific claim should still rest on the production-ready engine, not on the experimental search stack

This distinction matters a lot for rebuttal credibility.

### 1. Narrow the scope immediately

Do not keep selling this as a broad tabular-and-time-series framework in the rebuttal.

Safer framing:
- hierarchical tabular regression
- explicit contextual statistical features for domains with meaningful group structure
- evaluated on real-estate data only in the current submission

Avoid saying:
- broad tabular and time-series generality is already established
- model-agnostic in a strong universal sense
- the theory proves gains in full generality

### 2. Reposition novelty honestly

Do not argue that the primitive statistical operation is new.

The defendable novelty is:
- a formalized hierarchical representation layer, not a single encoding trick
- a leakage-safe multi-level statistical context system
- emphasis on systems of contextual summaries across hierarchy levels
- empirical evidence that coherent hierarchical context systems outperform the raw base representation on several real-world datasets

Best phrasing:
- "We agree that SCE is not a new primitive in the same sense as inventing target encoding. Our contribution is the formalization of hierarchical statistical context as a reusable representation layer, with leakage-safe construction, multi-level feature systems, and empirical analysis of when those systems help."

### 3. Tone down the theory claim

The current variance-reduction language is triggering skepticism because the features are target-derived.

Safer interpretation:
- this is not a universal theorem that any target-derived context must improve generalization
- it is an intuition for why cross-fitted group statistics can reduce effective prediction difficulty when hierarchical neighborhoods are informative and stable
- gains depend on leakage-safe construction, group quality, and downstream regularization

Best phrasing:
- "We will revise the theory section to make clear that the variance-reduction argument is an interpretation of why cross-fitted contextual statistics can help under structured heterogeneity, not a claim of unconditional improvement."

## Highest-ROI Response Themes

### Theme A: Stronger directly relevant baselines

This is the single most important gap.

Reviewers want comparison to:
- target encoding
- CatBoost-style ordered statistics
- hierarchical aggregation baselines that are close to SCE

If you can add only one new experiment block, add this one.

Minimum acceptable comparison set:
- base XGBoost / LightGBM
- standard target encoding on the same grouping keys
- hierarchical target mean only
- hierarchical mean + count only
- CatBoost baseline if feasible
- SCE full statistics system

The ideal point is not "we beat every baseline on every dataset".
The ideal point is to show that:
- SCE is stronger than plain target mean encoding
- gains do not come from a single cheap target statistic alone
- multi-statistic, multi-level context matters

### Theme B: Stability and uncertainty

RzPC explicitly asked for variability across seeds or folds.

Fastest useful additions:
- mean +/- std over repeated splits or repeated CV
- 95% confidence intervals or bootstrap intervals for RMSE deltas
- one short table showing stability, not just best numbers

If time is tight, do this on the two strongest datasets first.

### Theme C: Backoff and sparsity diagnostics

Reviewers are worried that the method may be surviving on near-identifiable tiny groups.

You need a small diagnostic table with:
- fraction of rows using native group stats vs backed-off stats
- distribution of effective group sizes
- fraction of groups below `min_group_size`
- performance vs backoff depth if available

Even a descriptive table helps a lot.

### Theme D: Claim narrowing on domain breadth

You probably cannot fully fix the cross-domain objection inside rebuttal unless you already have one clean extra dataset ready.

If you cannot add a new domain quickly:
- explicitly narrow the claim
- say the current paper demonstrates the method in one application family with strong natural hierarchies
- position broader validation as ongoing work, not established fact

### Theme E: Deep baselines

This matters, but it is probably not the highest-ROI rebuttal item unless you already have working scripts.

If you can add one, great.
If not, do not overinvest here before target-encoding baselines are done.

The current paper invited this criticism by contrasting against TabNet and TabTransformer too strongly.
In rebuttal:
- soften the contrast
- say current evidence supports complementarity and interpretability claims, not superiority to deep tabular models

## Administrative Risk: Blind Violation

Reviewer aRsy flagged a non-anonymous PyPI link.

This is serious because it is not a scientific criticism, it is a process issue.

What to do in the response:
- acknowledge the issue briefly and directly
- apologize without drama
- say no identifying artifact links are necessary for evaluating the scientific claims
- avoid repeating any identifying repository or package references in the rebuttal

Do not spend much rebuttal budget here.
One short acknowledgment is enough.

## What You Can Defend With Existing Material

Based on the current repo and paper draft, you can already defend:
- leakage-safe cross-fitting as a real construction choice, not hand-waving
- practical usefulness and ease of pipeline integration
- auto-detected grouping variables rather than mandatory hand-authored ordered hierarchies
- automatic interaction generation over grouping variables
- optional cleanup of leakage-like, correlated, and hierarchy-redundant features
- consistent gains across four real-world datasets
- interpretability of explicit context summaries
- ablation logic around systems of features rather than isolated single features

You cannot safely defend yet, at least not strongly:
- broad time-series generality
- broad cross-domain applicability
- superiority over the strongest target-encoding family baselines
- strong model-agnosticity across learner classes
- a very strong theoretical guarantee phrased as a general result
- fully validated claims about the experimental search/selection stack as if it were the core method
- robust hierarchical backoff analysis unless the empirical diagnostics are actually reported

## Priority Order for New Work

If rebuttal time is limited, do the items in this order.

1. Add target-encoding and CatBoost-adjacent baselines.
2. Add stability reporting: repeated splits, seed variance, or confidence intervals.
3. Add backoff and effective group-size diagnostics.
4. Narrow title, abstract, and contribution language.
5. Add one extra domain or one deep baseline only if the above is already covered.

## Suggested Rebuttal Structure

Use a short opening paragraph and then grouped responses by theme.

### Opening paragraph

Suggested direction:

"We thank the reviewers for the consistent and constructive feedback. We agree that the current draft overstates breadth relative to the present empirical scope. Our intended contribution is not a claim that SCE introduces a fundamentally new primitive beyond all forms of leakage-safe target aggregation, but rather that it formalizes hierarchical statistical context as a reusable representation layer with leakage-safe construction, interpretable multi-level features, and consistent gains in a practically important setting. In the revision we will narrow the scope to hierarchical tabular regression, clarify the theoretical interpretation, and add stronger directly relevant baselines and stability diagnostics."

### Response block 1: novelty and closest baselines

Key points to say:
- agree that SCE is related to target encoding and grouped aggregation
- distinction is systematic multi-level context representation, not a single target mean
- add strongest directly relevant baselines in revision or rebuttal update
- emphasize that the central empirical question is whether coherent multi-level statistical context systems outperform simpler aggregation baselines

### Response block 2: theory clarification

Key points to say:
- agree current wording is too strong
- clarify that contextual features are target-derived but cross-fitted
- revise theory as an explanatory lens for structured heterogeneity, not a universal guarantee
- tie benefit to stability of neighborhoods and leakage-safe construction

### Response block 3: robustness and reproducibility

Key points to say:
- report downstream model regularization settings clearly
- add repeated-split or fold-variance results
- add backoff frequency and effective group-size diagnostics
- clarify feature dimensionality management and no pre-final hidden selection if that is true

### Response block 4: scope and generality

Key points to say:
- agree the current empirical study is within one application family
- clarify that the implementation can auto-discover categorical grouping structure and interactions, so manual ordered hierarchy specification is not required in the default workflow
- still acknowledge that useful group structure must exist in the data for SCE to help
- revise claims accordingly
- avoid defending time-series breadth unless you actually add time-series evidence

### Response block 5: deep models

Key points to say:
- current submission does not establish superiority over deep tabular architectures
- revise text to present SCE as complementary and interpretable preprocessing
- if feasible, include at least one follow-up comparison

## Reviewer-Specific Notes

### Reviewer aRsy

Main answer:
- acknowledge blind issue
- agree novelty is incremental at primitive level
- defend formalization plus leakage-safe hierarchical system as the contribution
- if possible, mention sensitivity analysis on hierarchy choice will be added

### Reviewer RzPC

This is the most important reviewer to win over.

Answer directly in the order asked:
1. closest baselines
2. theory clarification
3. regularization and feature management
4. learner dependence
5. backoff frequency
6. domain breadth
7. sensitivity to hierarchy and fold count
8. why context-only underperforms despite importance dominance

For question 8, a good explanation is:
- base features identify the instance within the local distribution
- context features define the local distribution itself
- both are needed: context says what is normal here, base says where this sample sits inside that normal range

### Reviewer jNzT

This reviewer is receptive to practical value.

Best leverage:
- agree novelty is mainly in formalization and reusable system design
- say you will reduce the strength of novelty claims
- add or promise stronger matched baselines
- narrow generality claims

### Reviewer Tjq1

This reviewer reacted badly to the logic and explanation, not just to missing experiments.

Main fix:
- explain the mechanism more concretely
- state what problem is solved in one sentence: raw covariates often fail to represent local conditional distributions induced by hierarchy; SCE exposes those local distributions explicitly through leakage-safe statistical summaries
- do not rely only on metric tables; provide a causal story the reviewer can follow

## Tactical Recommendation

If you have limited bandwidth, do not spread yourself across too many new experiments.

The best rebuttal package is:
- one honest reframing paragraph
- one new baseline table against target-encoding family methods
- one stability table
- one backoff or group-size diagnostic table

That will do more than a rushed deep-learning comparison that is hard to trust.

## Next Concrete Actions Across Repos

### In `timeseries-context-engineering`
- update the paper draft language around scope and theory
- create a rebuttal note with exact response text
- add result tables for baseline family and uncertainty reporting

### In `sce`
- verify current implementation details for:
  - cross-fitting behavior
  - fallback logic
  - feature counts per dataset
  - model configuration and regularization reporting
- extract exact implementation details that can be cited in the rebuttal

## Bottom Line

The path to a credible rebuttal is not "our method is more novel than you think".

The credible path is:
- "you are right that the current draft overclaims"
- "here is the narrower and more accurate contribution"
- "here are stronger directly relevant baselines"
- "here is evidence that the gains are stable and not just tiny-group leakage artifacts"

That is the version of this submission that has the best chance of moving at least one reviewer.