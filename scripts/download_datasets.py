#!/usr/bin/env python3
"""
Compatibility wrapper around the manifest-driven dataset downloader.

Usage:
    python scripts/download_datasets.py
    python scripts/download_datasets.py --dataset rental_uae_contracts.parquet
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.download import main


if __name__ == "__main__":
    raise SystemExit(main())
