# Test configuration for equity package tests.
import sys
from pathlib import Path

# Add repo root first so tests resolve local `equity` package, not a site-packages install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
