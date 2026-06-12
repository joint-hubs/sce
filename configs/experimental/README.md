# Experimental dataset configs (excluded from the default set)

These configs are **not** part of the active benchmark set. They are excluded
from `sce.io.list_datasets()` and `python scripts/run.py --all` because they
currently fail the report-grade diagnostics gate (status as of 2026-06-12):

| Config | Why it is here |
|---|---|
| `rental_poland_long.toml` | Permuted-target and shuffled-groups diagnostics FAIL. Real SCE advantage (+1.22%) is within the noise floor (permuted target gives +3.1%) at n≈1000. Needs a simpler config or more data. |
| `sales_uae_transactions.toml` | Permuted-target diagnostic FAILS badly (+24.5% advantage on shuffled targets) on a 20k subsample — a memorization red flag. Needs a full-data or 100k+ rerun and a min_group_size review. |
| `rental_uae_contracts.toml` | SCE hurts performance (−6.7%) and shuffled-groups diagnostic fails on a 20k subsample of 5.5M rows. Likely a subsample artifact — rerun on full data. |

To run one anyway (diagnostics, debugging):

```bash
python scripts/run.py --dataset experimental/sales_uae_transactions
```

Re-promote a config back to `configs/` only after all four diagnostics pass
on report-grade (full-data) runs.
