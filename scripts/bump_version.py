#!/usr/bin/env python3
"""
Version bumping utility for SCE.

Usage:
    python scripts/bump_version.py patch   # 0.3.0 -> 0.3.1
    python scripts/bump_version.py minor   # 0.3.0 -> 0.4.0
    python scripts/bump_version.py major   # 0.3.0 -> 1.0.0
    python scripts/bump_version.py 0.4.0   # Set explicit version
"""

import re
import sys
from pathlib import Path


def get_current_version() -> str:
    """Read current version from sce/__init__.py."""
    init_path = Path(__file__).parent.parent / "sce" / "__init__.py"
    content = init_path.read_text()
    match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
    if not match:
        raise ValueError("Could not find __version__ in sce/__init__.py")
    return match.group(1)


def bump_version(current: str, bump_type: str) -> str:
    """Calculate new version based on bump type."""
    if bump_type not in ("major", "minor", "patch"):
        # Assume it's an explicit version
        return bump_type
    
    parts = current.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version format: {current}")
    
    major, minor, patch = map(int, parts)
    
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    else:  # patch
        return f"{major}.{minor}.{patch + 1}"


def update_file(path: Path, pattern: str, replacement: str) -> bool:
    """Update version in a file using regex replacement."""
    if not path.exists():
        print(f"  Skipping {path} (not found)")
        return False
    
    content = path.read_text()
    new_content = re.sub(pattern, replacement, content)
    
    if content == new_content:
        print(f"  No change in {path}")
        return False
    
    path.write_text(new_content)
    print(f"  Updated {path}")
    return True


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    bump_type = sys.argv[1]
    current_version = get_current_version()
    new_version = bump_version(current_version, bump_type)
    
    print(f"Bumping version: {current_version} -> {new_version}")
    
    root = Path(__file__).parent.parent
    
    # Update sce/__init__.py
    update_file(
        root / "sce" / "__init__.py",
        r'__version__\s*=\s*["\'][^"\']+["\']',
        f'__version__ = "{new_version}"'
    )
    
    # Update pyproject.toml
    update_file(
        root / "pyproject.toml",
        r'version\s*=\s*["\'][^"\']+["\']',
        f'version = "{new_version}"'
    )
    
    # Update CITATION.cff
    update_file(
        root / "CITATION.cff",
        r'version:\s*[^\n]+',
        f'version: {new_version}'
    )
    
    print(f"\nVersion bumped to {new_version}")
    print("\nNext steps:")
    print(f"  1. Update CHANGELOG.md with release notes")
    print(f"  2. Commit: git commit -am 'Bump version to {new_version}'")
    print(f"  3. Tag: git tag v{new_version}")
    print(f"  4. Push: git push origin main --tags")


if __name__ == "__main__":
    main()
