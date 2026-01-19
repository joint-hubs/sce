#!/usr/bin/env python3
"""
CI script for generating architecture diagrams from code metadata.

Wraps the autodocs generator for CI integration.

Parses @module headers and @component decorators to auto-generate:
- Dependency graphs (Mermaid)
- C4 architecture diagrams
- Module documentation
- README injection

Usage:
    python scripts/ci/generate_docs.py [--output=docs/generated/]
"""

import sys
from pathlib import Path

# Ensure we can import autodocs
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from autodocs.parser import scan_project
from autodocs.generators.mermaid import generate_all_diagrams
from autodocs.generators.markdown import generate_api_reference
from autodocs.generators.injector import process_readme


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate documentation from code metadata"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/generated"),
        help="Output directory"
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=Path("README.md"),
        help="README to inject into"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("sce"),
        help="Source path to scan"
    )
    parser.add_argument(
        "--skip-inject",
        action="store_true",
        help="Skip README injection"
    )
    
    args = parser.parse_args()
    
    print("🔍 Scanning project...")
    project = scan_project(args.path)
    print(f"   Found {len(project.modules)} modules, {len(project.components)} components")
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Generate diagrams
    print("\n📊 Generating Mermaid diagrams...")
    diagrams = generate_all_diagrams(project)
    
    for name, content in diagrams.items():
        out_path = args.output / f"{name}.md"
        out_path.write_text(content, encoding='utf-8')
        print(f"   ✅ {out_path}")
    
    # Generate API reference
    print("\n📝 Generating API reference...")
    api_ref = generate_api_reference(project)
    api_path = args.output / "api_reference.md"
    api_path.write_text(api_ref, encoding='utf-8')
    print(f"   ✅ {api_path}")
    
    # Inject into README
    if not args.skip_inject and args.readme.exists():
        print(f"\n📥 Injecting into {args.readme}...")
        results = process_readme(project, args.readme, backup=False)
        for marker, updated in results.items():
            status = "✅" if updated else "⏭️"
            print(f"   {status} {marker}")
    
    print("\n✅ Documentation generated!")


if __name__ == "__main__":
    main()

