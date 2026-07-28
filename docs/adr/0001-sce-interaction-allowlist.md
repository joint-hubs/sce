# ADR 0001: SCE interaction allow-list

**Status:** Proposed

**Date:** 2026-07-28

**Linear:** FOC-51 (S4.3)

## Context

Upstream `sce.ContextConfig.include_interactions` is a **boolean only**
(`sce/config.py` ~line 236 / 238). When `True`,
`sce.stats.compute_aggregations` expands **all** k-tuples of
`categorical_cols` up to `max_interaction_depth`
(`sce/stats.py:271-292`):

```python
for depth in range(2, min(max_interaction_depth + 1, len(all_categoricals) + 1)):
    for combo in combinations(all_categoricals, depth):
        level_name = "__".join(group_cols)  # e.g. "sector__time_bucket"
```

There is no way to pass an **explicit subset** of interactions. For the
equity manifold that is a problem:

| Pair | Useful? | Why |
|---|---|---|
| `sector × time_bucket` | yes | sector regime by calendar month |
| `industry × time_bucket` | yes | finer GICS regime |
| `mktcap_bucket × time_bucket` | yes | size factor regime |
| `sector × mktcap_bucket` | yes | sector tilt by size |
| `ticker × industry` | no | each ticker already is its own group; industry is stable |
| `ticker × sector` | no | redundant with ticker leaf + sector leaf |
| `ticker × time_bucket` | no | explodes cardinality (ticker×month) and is sparse |

`ticker` is a **leaf group key** (single-col level). It must stay in
`categorical_cols` so stats are isolated per-ticker, but it must **not**
enter interaction pairs.

The S4 equity enrichment pipeline (`equity.sce`) therefore needs a curated
allow-list without forking or patching the vendored `sce/` tree (P0: do
not modify `sce/`).

## Decision

### Upstream proposal (this ADR's primary decision)

Add an optional field to `ContextConfig`:

```python
interactions: Optional[List[Tuple[str, ...]]] = None
```

Semantics:

* `None` (default) — current behavior: build **all** k-tuples of
  `categorical_cols` up to `max_interaction_depth`. Fully
  backward-compatible.
* Non-`None` — build **only** the listed combinations (plus the
  always-on single-col levels + `global`). Each tuple is an ordered
  combination of column names already present in `categorical_cols`.
  `max_interaction_depth` still bounds the longest allowed tuple
  (reject / ignore longer entries with a clear error/warning).

The level name for a listed interaction remains
`"__".join(tuple)` so downstream column naming
`{level}_{target}_{stat}` is unchanged.

### Interim (S4 NOW, before upstream lands)

The equity wrapper (`equity.sce.EquityContextEnricher`) **post-filters
SCE OUTPUT columns**:

1. Run SCE with `include_interactions=True` (it still builds every pair).
2. Derive each added column's level name by stripping the trailing
   `_{target_col}_{stat}` suffix (see unit tests / enricher parser).
3. Drop any column whose level:
   * contains `"__"` (interaction), **and**
   * is **not** in the equity allow-list
     (`EquityHierarchyConfig.interactions`).
4. Keep all single-col levels (`ticker`, `sector`, ...), `global`, and
   allow-listed interaction levels.

This bounds effective cardinality for S4 without touching `sce/`.

Default equity allow-list (`DEFAULT_INTERACTIONS`):

```python
(
    ("sector", "time_bucket"),
    ("industry", "time_bucket"),
    ("mktcap_bucket", "time_bucket"),
    ("sector", "mktcap_bucket"),
)
```

## Consequences

### Positive

* Equity S4 unblocked without a vendored-code fork.
* Clear, reviewable contract for the eventual upstream field
  (`interactions: Optional[List[Tuple[str, ...]]]`).
* Default `None` keeps existing SCE users sesion-compatible.

### Negative / costs

* **Post-filter wastes compute**: SCE still builds every pair (including
  `ticker__*`), then equity drops them. Acceptable for the 33-ticker
  seed; becomes costly at full S&P 500 × long history.
* Naming parser is dual code: equity must mirror SCE's
  `{level}_{target}_{stat}` convention. A future rename upstream would
  desync the filter (covered by unit tests).
* Fold-variance suffixes (`_fold_std` etc.) must be stripped with the
  same level parser when random CF is used (rolling CF currently does
  not emit them).

### Neutral

* Once the upstream field lands, the equity post-filter becomes a
  no-op / is deleted and `ContextConfig.interactions` is set directly
  from `EquityHierarchyConfig.interactions`.

## Alternatives

1. **`List[Tuple[str, str]]` pairs-only field.**
   Rejected: no forward-compat for triples if
   `max_interaction_depth` rises; `Tuple[str, ...]` is the general form.

2. **Run SCE once per allow-listed pair and outer-merge.**
   Rejected: N fits, duplicated global/single-col stats, ugly join
   bookkeeping, worse than a post-filter for S4.

3. **`Iterable[Tuple[str, ...]]` instead of `List`.**
   Noted as an equivalent general form; `List` preferred for
   dataclass serialization / CLI friendliness. Not a real alternative.

4. **Monkey-patch / fork `sce/stats.py`.**
   Rejected hard: P0 constraint is that `sce/` is vendored upstream and
   must not be modified from the equity layer.

## References

* PRD: `docs/plan/2026-07-27_trading_forecaster_prd.md` §6.1 (SCE
  hierarchy / no-lookahead).
* Upstream config: `sce/config.py` (`ContextConfig`,
  `include_interactions`, `max_interaction_depth`).
* Upstream expansion: `sce/stats.py:271-292`
  (`compute_aggregations` interaction loop).
* Upstream join naming: `sce/engine.py:713-734`
  (`_join_level_stats` → `{level_name}_{stat_col}` where
  `stat_col = {target}_{method}`).
* Equity consumer: `equity/sce/` (this slice, FOC-51 S4.1/S4.2).
