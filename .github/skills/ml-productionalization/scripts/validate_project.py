#!/usr/bin/env python3
"""
Project Health Validator for ML Productionalization

Scans an ML project for common issues that would prevent production deployment.
Returns a report with actionable fixes.

Usage:
    python validate_project.py --path /path/to/project
    python validate_project.py --path . --verbose
    python validate_project.py --path . --check reproducibility
"""

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

# ═══════════════════════════════════════════════════════════════════════════════
# ISSUE DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class Issue:
    """A single issue found during validation."""

    category: str
    severity: str  # "error", "warning", "info"
    file: Path | None
    line: int | None
    message: str
    fix: str

    def __str__(self) -> str:
        location = ""
        if self.file:
            location = f"{self.file}"
            if self.line:
                location += f":{self.line}"
            location += " - "
        return f"[{self.severity.upper()}] {location}{self.message}"


@dataclass
class ValidationReport:
    """Aggregated validation results."""

    project_path: Path
    issues: list[Issue] = field(default_factory=list)
    checks_passed: list[str] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def is_healthy(self) -> bool:
        return self.error_count == 0

    def print_summary(self) -> None:
        print("\n" + "═" * 60)
        print("PROJECT HEALTH REPORT")
        print("═" * 60)

        if self.checks_passed:
            print("\n✅ Passed Checks:")
            for check in self.checks_passed:
                print(f"   • {check}")

        if self.issues:
            print(f"\n❌ Issues Found: {len(self.issues)}")
            print(f"   Errors: {self.error_count}, Warnings: {self.warning_count}")

            # Group by category
            by_category: dict[str, list[Issue]] = {}
            for issue in self.issues:
                by_category.setdefault(issue.category, []).append(issue)

            for category, issues in by_category.items():
                print(f"\n   [{category.upper()}]")
                for issue in issues:
                    print(f"   {issue}")
                    print(f"      Fix: {issue.fix}")

        print("\n" + "═" * 60)
        if self.is_healthy:
            print("✅ PROJECT IS PRODUCTION-READY")
        else:
            print("❌ PROJECT NEEDS ATTENTION")
        print("═" * 60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKERS
# ═══════════════════════════════════════════════════════════════════════════════


class BaseChecker:
    """Base class for project health checkers."""

    name: str = "base"

    def __init__(self, project_path: Path):
        self.project_path = project_path

    def check(self) -> Iterator[Issue]:
        """Yield issues found. Override in subclasses."""
        raise NotImplementedError


class DependencyChecker(BaseChecker):
    """Checks dependency management for reproducibility."""

    name = "dependencies"

    def check(self) -> Iterator[Issue]:
        # Check for requirements.txt or pyproject.toml
        has_requirements = (self.project_path / "requirements.txt").exists()
        has_pyproject = (self.project_path / "pyproject.toml").exists()

        if not has_requirements and not has_pyproject:
            yield Issue(
                category=self.name,
                severity="error",
                file=None,
                line=None,
                message="No dependency file found",
                fix="Create requirements.txt or pyproject.toml with pinned versions",
            )
            return

        # Check for pinned versions in requirements.txt
        if has_requirements:
            req_path = self.project_path / "requirements.txt"
            with open(req_path) as f:
                for i, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    # Check if version is pinned
                    if "==" not in line and ">=" in line:
                        yield Issue(
                            category=self.name,
                            severity="warning",
                            file=req_path,
                            line=i,
                            message=f"Unpinned dependency: {line}",
                            fix="Use == instead of >= for reproducibility",
                        )

        # Check for lock file
        has_lock = any(
            (self.project_path / f).exists()
            for f in [
                "requirements-lock.txt",
                "poetry.lock",
                "Pipfile.lock",
                "pdm.lock",
            ]
        )
        if not has_lock:
            yield Issue(
                category=self.name,
                severity="warning",
                file=None,
                line=None,
                message="No lock file found",
                fix="Generate with: pip freeze > requirements-lock.txt",
            )


class ReproducibilityChecker(BaseChecker):
    """Checks for reproducibility issues like missing seeds."""

    name = "reproducibility"

    # Patterns that indicate random operations without seeds
    RANDOM_PATTERNS = [
        (r"np\.random\.(rand|randn|randint|choice|shuffle)\(", "np.random.seed()"),
        (r"random\.(random|randint|choice|shuffle)\(", "random.seed()"),
        (r"\.sample\((?!.*random_state)", "Add random_state= parameter"),
        (r"train_test_split\((?!.*random_state)", "Add random_state= parameter"),
        (r"KFold\((?!.*random_state)", "Add random_state= parameter"),
    ]

    def check(self) -> Iterator[Issue]:
        for py_file in self._find_python_files():
            yield from self._check_file(py_file)

    def _find_python_files(self) -> Iterator[Path]:
        """Find all Python files, excluding venv/hidden directories."""
        for pattern in ["**/*.py"]:
            for path in self.project_path.glob(pattern):
                # Skip virtual environments and hidden directories
                parts = path.relative_to(self.project_path).parts
                if any(
                    p.startswith(".") or p in ("venv", ".venv", "__pycache__", "node_modules")
                    for p in parts
                ):
                    continue
                yield path

    def _check_file(self, path: Path) -> Iterator[Issue]:
        """Check a single file for reproducibility issues."""
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern, fix in self.RANDOM_PATTERNS:
                if re.search(pattern, line):
                    yield Issue(
                        category=self.name,
                        severity="warning",
                        file=path,
                        line=i,
                        message=f"Potential unseeded random operation: {line.strip()[:50]}...",
                        fix=fix,
                    )


class HardcodedPathChecker(BaseChecker):
    """Checks for hardcoded paths that break portability."""

    name = "portability"

    # Patterns for common hardcoded paths
    HARDCODED_PATTERNS = [
        r'["\']C:\\',
        r'["\']D:\\',
        r'["\']/home/',
        r'["\']/Users/',
        r'["\']/mnt/',
    ]

    def check(self) -> Iterator[Issue]:
        for py_file in self._find_python_files():
            yield from self._check_file(py_file)

    def _find_python_files(self) -> Iterator[Path]:
        """Find all Python files."""
        for path in self.project_path.glob("**/*.py"):
            parts = path.relative_to(self.project_path).parts
            if any(
                p.startswith(".") or p in ("venv", ".venv", "__pycache__")
                for p in parts
            ):
                continue
            yield path

    def _check_file(self, path: Path) -> Iterator[Issue]:
        """Check a single file for hardcoded paths."""
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return

        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            for pattern in self.HARDCODED_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    yield Issue(
                        category=self.name,
                        severity="error",
                        file=path,
                        line=i,
                        message=f"Hardcoded path: {line.strip()[:60]}...",
                        fix="Use Path.cwd(), environment variables, or config files",
                    )


class DocumentationChecker(BaseChecker):
    """Checks for missing or incomplete documentation."""

    name = "documentation"

    def check(self) -> Iterator[Issue]:
        # Check for README
        readme_patterns = ["README.md", "README.rst", "README.txt", "README"]
        has_readme = any((self.project_path / p).exists() for p in readme_patterns)
        if not has_readme:
            yield Issue(
                category=self.name,
                severity="error",
                file=None,
                line=None,
                message="No README file found",
                fix="Create README.md with project overview, installation, and usage",
            )

        # Check for docstrings in main modules
        src_dirs = ["src", "lib", "."]
        for src_dir in src_dirs:
            src_path = self.project_path / src_dir
            if src_path.exists():
                for py_file in src_path.glob("*.py"):
                    if py_file.name.startswith("_"):
                        continue
                    yield from self._check_docstrings(py_file)

    def _check_docstrings(self, path: Path) -> Iterator[Issue]:
        """Check for missing docstrings in public functions/classes."""
        try:
            content = path.read_text(encoding="utf-8")
            tree = ast.parse(content)
        except (SyntaxError, UnicodeDecodeError):
            return

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                # Skip private functions
                if node.name.startswith("_"):
                    continue

                # Check for docstring
                docstring = ast.get_docstring(node)
                if not docstring:
                    yield Issue(
                        category=self.name,
                        severity="warning",
                        file=path,
                        line=node.lineno,
                        message=f"Missing docstring: {node.name}",
                        fix="Add a docstring explaining purpose, args, and returns",
                    )


class TestingChecker(BaseChecker):
    """Checks for testing infrastructure."""

    name = "testing"

    def check(self) -> Iterator[Issue]:
        # Check for tests directory
        test_dirs = ["tests", "test", "testing"]
        has_tests = any((self.project_path / d).is_dir() for d in test_dirs)

        if not has_tests:
            yield Issue(
                category=self.name,
                severity="error",
                file=None,
                line=None,
                message="No tests directory found",
                fix="Create tests/ directory with pytest tests",
            )
            return

        # Check for pytest.ini or pyproject.toml [tool.pytest]
        has_pytest_config = (self.project_path / "pytest.ini").exists()
        if not has_pytest_config:
            pyproject = self.project_path / "pyproject.toml"
            if pyproject.exists():
                content = pyproject.read_text()
                has_pytest_config = "[tool.pytest" in content

        if not has_pytest_config:
            yield Issue(
                category=self.name,
                severity="info",
                file=None,
                line=None,
                message="No pytest configuration found",
                fix="Add pytest config to pyproject.toml for consistent test runs",
            )

        # Count test files
        test_count = 0
        for test_dir in test_dirs:
            test_path = self.project_path / test_dir
            if test_path.is_dir():
                test_count += len(list(test_path.glob("**/test_*.py")))
                test_count += len(list(test_path.glob("**/*_test.py")))

        if test_count == 0:
            yield Issue(
                category=self.name,
                severity="error",
                file=None,
                line=None,
                message="No test files found (test_*.py or *_test.py)",
                fix="Add tests in tests/ directory",
            )
        elif test_count < 5:
            yield Issue(
                category=self.name,
                severity="warning",
                file=None,
                line=None,
                message=f"Only {test_count} test file(s) found",
                fix="Consider adding more tests for better coverage",
            )


class ConfigChecker(BaseChecker):
    """Checks configuration files for completeness."""

    name = "configuration"

    def check(self) -> Iterator[Issue]:
        # Check for config directory
        config_dirs = ["configs", "config", "conf"]
        config_path = None
        for d in config_dirs:
            if (self.project_path / d).is_dir():
                config_path = self.project_path / d
                break

        if config_path:
            # Check for documented configs
            config_files = list(config_path.glob("**/*.toml")) + list(
                config_path.glob("**/*.yaml")
            )
            for config_file in config_files:
                yield from self._check_config_file(config_file)

    def _check_config_file(self, path: Path) -> Iterator[Issue]:
        """Check a config file for documentation."""
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        # Count comment lines vs total lines
        comment_lines = sum(1 for line in lines if line.strip().startswith("#"))
        total_lines = sum(1 for line in lines if line.strip())

        if total_lines > 10 and comment_lines / total_lines < 0.1:
            yield Issue(
                category=self.name,
                severity="warning",
                file=path,
                line=None,
                message="Config file has minimal comments (<10%)",
                fix="Add comments explaining each configuration option",
            )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


ALL_CHECKERS = [
    DependencyChecker,
    ReproducibilityChecker,
    HardcodedPathChecker,
    DocumentationChecker,
    TestingChecker,
    ConfigChecker,
]


def validate_project(
    project_path: Path, checks: list[str] | None = None
) -> ValidationReport:
    """
    Validate a project and return a report.

    Args:
        project_path: Path to the project root
        checks: Optional list of specific checks to run (e.g., ["dependencies", "testing"])

    Returns:
        ValidationReport with all issues found
    """
    report = ValidationReport(project_path=project_path)

    for checker_class in ALL_CHECKERS:
        # Filter by check name if specified
        if checks and checker_class.name not in checks:
            continue

        checker = checker_class(project_path)
        issues = list(checker.check())

        if issues:
            report.issues.extend(issues)
        else:
            report.checks_passed.append(checker.name)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate ML project for production readiness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s --path .
    %(prog)s --path /path/to/project --verbose
    %(prog)s --path . --check dependencies --check testing
    
Available checks:
    dependencies    - Pinned versions, lock files
    reproducibility - Random seeds, deterministic operations
    portability     - No hardcoded paths
    documentation   - README, docstrings
    testing         - Test infrastructure
    configuration   - Config file documentation
        """,
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path.cwd(),
        help="Path to project root (default: current directory)",
    )
    parser.add_argument(
        "--check",
        action="append",
        dest="checks",
        help="Specific check(s) to run (can be repeated)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed output"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    args = parser.parse_args()

    if not args.path.exists():
        print(f"Error: Path does not exist: {args.path}", file=sys.stderr)
        sys.exit(1)

    report = validate_project(args.path, args.checks)

    if args.json:
        output = {
            "project": str(report.project_path),
            "is_healthy": report.is_healthy,
            "error_count": report.error_count,
            "warning_count": report.warning_count,
            "checks_passed": report.checks_passed,
            "issues": [
                {
                    "category": i.category,
                    "severity": i.severity,
                    "file": str(i.file) if i.file else None,
                    "line": i.line,
                    "message": i.message,
                    "fix": i.fix,
                }
                for i in report.issues
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        report.print_summary()

    sys.exit(0 if report.is_healthy else 1)


if __name__ == "__main__":
    main()
