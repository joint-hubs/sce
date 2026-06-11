# Copilot Instructions for SCE Project

**Repository:** [github.com/joint-hubs/sce](https://github.com/joint-hubs/sce)

> **Golden Rule**: Every time you learn something new or achieve something, update the relevant skill, create a reference, or document the learning. Knowledge that stays in your head dies with the session.

---

## Core Principles

### 1. Learn → Document → Share

When you:
- Discover a bug or edge case → Add it to the relevant skill's "Gotchas" section
- Figure out how something works → Update the CODE_MAP or create a reference
- Solve a problem → Document the solution pattern
- Find a gap between paper and code → Log it in sce-algorithm skill

**Never leave knowledge undocumented.**

### 2. Skills Are Living Documents

Skills are not static. They evolve as the project evolves.

| When this happens... | Do this... |
|---|---|
| You learn a new pattern | Add to skill's "Common Patterns" |
| You hit a blocker | Add to "Gotchas" |
| You create a useful script | Add to skill's `scripts/` folder |
| You find missing docs | Create a reference in `references/` |
| A procedure changes | Update the skill's "Procedure" section |

### 3. Before You Start Any Task

1. Check if a relevant skill exists in `.github/skills/`
2. Read the SKILL.md to understand context
3. Check references for detailed information
4. If no skill exists and the task is complex, consider creating one

### 4. After You Complete Any Task

Ask yourself:
- Did I learn something that would help future agents?
- Did I discover a gap in documentation?
- Did I solve a problem that was hard to figure out?

If yes to any → **Update the relevant skill or create a new reference.**

---

## Skill Discovery

Skills are located in `.github/skills/`. Each skill has:
- `SKILL.md` — Quick reference and procedures
- `references/` — Detailed documentation
- `scripts/` — Automation helpers (optional)

### Available Skills

| Skill | Use When |
|---|---|
| `sce-algorithm` | Verifying code matches paper, checking leakage |
| `sce-library-design` | Creating modules, checking architecture |
| `sce-data` | Handling datasets, anonymization |
| `sce-experiments` | Running experiments, reproducing results |
| `paper-publication` | Generating figures, tables for paper |
| `ci-autodocs` | Setting up CI, auto-documentation |
| `ml-productionalization` | Shipping code, testing, reproducibility |

---

## Specialized Agents

Agents are specialized roles defined as `.agent.md` files in `.github/agents/`. See `.github/agents/AGENTS.md` for full details.

| Agent | Purpose | Status |
|---|---|---|
| `auditor.agent.md` | Verify code matches paper equations | ✅ Ready |
| `experimenter.agent.md` | Run and validate experiments | ✅ Ready |
| `publisher` | Generate paper figures/tables | ⏸️ Deferred |
| `architect` | Enforce architecture standards | ⏸️ Deferred |
| `shipper` | Prepare release | ⏸️ Deferred |

To activate an agent, select it from the VS Code Chat agents dropdown or use `@agent-name` syntax.

---

## Code Standards

### Module Headers (Required)

Every Python module must have a metadata header:

```python
"""
@module: sce.engine
@depends: sce.config, sce.stats
@exports: StatisticalContextEngine
@paper_ref: Algorithm 1, Eq. 1-4
"""
```

See `docs/standards/code_metadata.md` for full spec.

### File Size Limit

- **Max 300 lines per file**
- Split by responsibility when growing

### Dependency Rules

- See `.github/skills/sce-library-design/references/DEPENDENCY_RULES.md`
- CI will validate imports


## Critical Reminders

1. **Paper accuracy matters**: The code must match the paper exactly. Any deviation must be documented and justified.

2. **No source brand names**: Use anonymized dataset names (see sce-data skill).

3. **Cross-fitting is implemented**: The code uses 5-fold cross-fitting for leakage-safe context features. Fold variance features (`_fold_std`, `_fold_lower`, `_fold_upper`) capture uncertainty across folds.

4. **Test before you ship**: Use the ml-productionalization skill for release prep.

5. **Visualization feature classification**: When generating figures, ensure SCE features (with suffixes like `_mean`, `_fold_lower`, etc.) are correctly classified using `is_sce_feature()`.

---

## Current Results (2026-01-22)

| Dataset | Baseline RMSE | SCE RMSE | ΔRMSE | ΔR² |
|---------|---------------|----------|-------|-----|
| rental_poland_long | 4,581 | 4,541 | ↓ 0.9% | +1.55 pp |
| rental_poland_short | 27,368 | 22,541 | ↓ 17.6% | +24.49 pp |
| rental_uae_contracts | 465,037 | 360,267 | ↓ 22.5% | +3.83 pp |
| sales_uae_transactions | 32,489,660 | 26,353,228 | ↓ 18.9% | +25.83 pp |

---

## When You Don't Know Something

1. Search existing skills and references
2. Check the paper (`paper_overleaf_format.txt`)
3. Read relevant source code
4. If still unclear, document the question in the relevant skill as a TODO

---

## The Meta-Rule

> If you find yourself thinking "someone should document this" — that someone is you, and the time is now.

Knowledge compounds. Every small documentation update makes the next agent faster and more effective.
