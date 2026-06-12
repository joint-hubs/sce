"""
@module: sce.cli
@depends: sce.io
@exports: main
@paper_ref: N/A
@data_flow: CLI args -> dataset utilities
"""

from __future__ import annotations

import argparse

from sce.io import ensure_dataset, get_dataset_info, list_datasets, verify_all_datasets


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SCE command-line utilities")
    subparsers = parser.add_subparsers(dest="command")

    datasets_parser = subparsers.add_parser("datasets", help="Dataset utilities")
    datasets_subparsers = datasets_parser.add_subparsers(dest="datasets_command")

    datasets_subparsers.add_parser("list", help="List available datasets")

    info_parser = datasets_subparsers.add_parser("info", help="Show dataset metadata")
    info_parser.add_argument("dataset", help="Dataset name or config path")

    download_parser = datasets_subparsers.add_parser(
        "download", help="Download a dataset if it is configured as remote"
    )
    download_parser.add_argument("dataset", help="Dataset name or config path")
    download_parser.add_argument("--force", action="store_true", help="Force re-download")

    datasets_subparsers.add_parser("verify", help="Verify all configured datasets")
    return parser


def main() -> int:
    """Command-line interface for SCE dataset utilities."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command != "datasets":
        parser.print_help()
        return 0

    if args.datasets_command == "list":
        for info in list_datasets():
            location = "local" if info.exists_locally else "missing"
            if info.remote_source is not None:
                remote = info.remote_source
            elif info.source == "generated":
                remote = "generated-local-build"
            else:
                remote = "bundled"
            print(f"{info.name}: {location} | source={remote}")
        return 0

    if args.datasets_command == "info":
        info = get_dataset_info(args.dataset)
        remote_source = info.remote_source
        if remote_source is None:
            remote_source = "generated-local-build" if info.source == "generated" else "bundled"
        print(f"name: {info.name}")
        print(f"path: {info.path}")
        print(f"description: {info.description}")
        print(f"source: {info.source}")
        print(f"remote_source: {remote_source}")
        print(f"exists_locally: {info.exists_locally}")
        if info.size_bytes is not None:
            print(f"size_bytes: {info.size_bytes}")
        return 0

    if args.datasets_command == "download":
        path = ensure_dataset(args.dataset, force_download=args.force)
        print(path)
        return 0

    if args.datasets_command == "verify":
        results = verify_all_datasets()
        for name, is_valid in results.items():
            status = "OK" if is_valid else "INVALID"
            print(f"{name}: {status}")
        return 0 if all(results.values()) else 1

    datasets_parser = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    datasets_parser.choices["datasets"].print_help()
    return 0


__all__ = ["main"]
