"""Feature dominance audit CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.run import audit_feature_dominance


def audit_feature_dominance_file(
    importance_csv: Path,
    top_k: int = 3,
    threshold_pct: float = 70.0,
) -> dict:
    return audit_feature_dominance(importance_csv, top_k=top_k, threshold_pct=threshold_pct)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit feature-importance dominance")
    parser.add_argument("--importance-csv", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--threshold-pct", type=float, default=70.0)
    args = parser.parse_args()

    result = audit_feature_dominance_file(
        Path(args.importance_csv),
        top_k=args.top_k,
        threshold_pct=args.threshold_pct,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
