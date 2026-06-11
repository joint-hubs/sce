# Dataset Map (Anonymized)

| Production Name | Domain | Storage | Format |
|---|---|---|---|
| `rental_poland_short` | Short-term rental listings (PL) | In-repo | Parquet |
| `rental_poland_long` | Long-term rental listings (PL) | In-repo | Parquet |
| `rental_uae_contracts` | Rental contracts (UAE) | Remote | Parquet |
| `sales_uae_transactions` | Property sales (UAE) | Remote | Parquet |

## Notes

- UAE datasets should be hosted on a free public storage (e.g., Hugging Face Datasets).
- Small PL datasets can be committed after anonymization and PII removal.
