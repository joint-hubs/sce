#!/usr/bin/env python3
"""
CI script for validating code metadata compliance.

Wraps the autodocs validator for CI integration.

Usage:
    python scripts/ci/validate_metadata.py [--strict]
    
Checks:
- All modules have required @module headers
- @depends lists only internal packages
- @exports matches actual public symbols
- No circular dependencies
"""

import sys
from pathlib import Path

# Ensure we can import autodocs
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autodocs.parser import scan_project
from autodocs.validator import validate_project


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate code metadata for CI"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("sce"),
        help="Path to scan (default: sce)"
    )
    
    args = parser.parse_args()
    
    print("🔍 Scanning project for metadata...")
    project = scan_project(args.path)
    print(f"   Found {len(project.modules)} modules, {len(project.components)} components")
    
    print("\n📋 Validating metadata...")
    success, report = validate_project(project, strict=args.strict)
    
    print(report)
    
    if success:
        print("\n✅ Validation passed!")
        sys.exit(0)
    else:
        print("\n❌ Validation failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()

