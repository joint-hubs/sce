# ML Productionalization Skill Set — Index

> **Purpose**: Transform an R&D ML project into a production-ready, reproducible, well-tested codebase that looks like a talented human wrote it.

---

## Target Project

**Statistical Context Engineering (SCE)** — A Python ML pipeline that enriches regression features with hierarchical statistical context.

---

## Skill Files Overview

| # | File | Purpose | Status |
|---|------|---------|--------|
| 1 | [SKILL.md](SKILL.md) | Master skill — triggers, philosophy, workflow overview | ✅ Complete |
| 2 | [THE_CRAFT.md](THE_CRAFT.md) | Code style philosophy — writing code that tells a story | ✅ Complete |
| 3 | [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md) | Testing strategy for ML pipelines — property-based, fixtures, edge cases | ✅ Complete |
| 4 | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Ensuring experiments are reproducible — seeds, versioning, artifacts | ✅ Complete |
| 5 | [DOCUMENTATION_STYLE.md](DOCUMENTATION_STYLE.md) | Writing docs that don't look AI-generated — voice, narrative, examples | ✅ Complete |
| 6 | [CODE_REVIEW.md](CODE_REVIEW.md) | Self-review checklist for production readiness | ✅ Complete |
| 7 | [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) | Final pre-release validation — CI, packaging, changelog | ✅ Complete |
| 8 | [scripts/validate_project.py](scripts/validate_project.py) | Automated validation script for project health | ✅ Complete |

---

## Creation Order

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SKILL CREATION SEQUENCE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   1. SKILL.md           → Entry point, activates the whole set      │
│   2. THE_CRAFT.md       → Philosophy before tactics                 │
│   3. TESTING_WORKFLOW   → Foundation of reliability                 │
│   4. REPRODUCIBILITY    → Core ML engineering concern               │
│   5. DOCUMENTATION      → Make it human, not robotic                │
│   6. CODE_REVIEW        → Self-check before others see it           │
│   7. RELEASE_CHECKLIST  → Final gates before shipping               │
│   8. validate_project.py → Automation to enforce it all             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Design Principles

### 1. Story Over Optimization
Code should read like a narrative. Variable names, function flow, and comments should tell the *why*, not just the *what*.

### 2. Opinionated But Justified
Every decision has a rationale. No cargo-cult patterns. If we do something, we know why.

### 3. Human Fingerprints
AI-written code often has a sterile quality — too consistent, too generic. We want:
- Personality in comments
- Pragmatic shortcuts where they make sense
- Occasional "this is ugly but works" honesty

### 4. Reproducibility as Religion
Same inputs → same outputs. Every time. On any machine. No exceptions.

### 5. Tests That Prove Something
Not just "assert True". Tests should demonstrate properties, catch regressions, and document behavior.

---

## Quick Navigation

When you need to:

| Task | Read This |
|------|-----------|
| Understand the philosophy | [THE_CRAFT.md](THE_CRAFT.md) |
| Add tests to the project | [TESTING_WORKFLOW.md](TESTING_WORKFLOW.md) |
| Make experiments reproducible | [REPRODUCIBILITY.md](REPRODUCIBILITY.md) |
| Write non-generic docs | [DOCUMENTATION_STYLE.md](DOCUMENTATION_STYLE.md) |
| Review your own code | [CODE_REVIEW.md](CODE_REVIEW.md) |
| Prepare for release | [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) |
| Run automated checks | [scripts/validate_project.py](scripts/validate_project.py) |

---

## Integration with Fenix

This skill set is designed for standalone use but follows Fenix patterns:

- Lives in `.github/skills/ml-productionalization/`
- Uses progressive disclosure (load what you need)
- Can be referenced by other skills or agents

---

*"The goal is not to write code that works. The goal is to write code that a talented engineer would be proud to show."*
