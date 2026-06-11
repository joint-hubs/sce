---
name: sce-data
description: SCE dataset management, loading, and anonymization. Use when loading datasets, anonymizing sources, managing parquet files, choosing storage strategy, or validating checksums. Triggers include dataset, parquet, download, anonymize, checksums, Hugging Face.
---

# SCE Data Handling

## Quick Reference

| Document | Purpose |
|---|---|
| references/DATASETS.md | Anonymized dataset map |
| references/ANONYMIZATION.md | Naming rules |

## When to Use This Skill

- You need to add or rename datasets
- You need to define storage strategy (repo vs remote)
- You need to document data access for CI

## Procedure

1. Use anonymized names from DATASETS.md.
2. Store small datasets in repo as parquet.
3. Host large datasets remotely and provide a download script.
4. Verify checksums before use.

## Gotchas

- Do not mention source brand names in public docs or configs.
- Do not commit large datasets to git history.
- rental_poland_long has `development_floor` fully missing; exclude it from numeric features or it will drop all rows.
