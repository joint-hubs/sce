"""AST guard for unsafe engine.fit_transform usage outside approved scope."""

from __future__ import annotations

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = [PROJECT_ROOT / "sce", PROJECT_ROOT / "scripts"]


class FitTransformCallVisitor(ast.NodeVisitor):
    """Collect suspicious engine.fit_transform calls."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.function_stack: list[str] = []
        self.violations: list[str] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_stack.append(node.name)
        self.generic_visit(node)
        self.function_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "fit_transform":
            receiver = node.func.value
            if isinstance(receiver, ast.Name) and receiver.id == "engine":
                if not self._is_allowed_location():
                    rel = self.file_path.relative_to(PROJECT_ROOT).as_posix()
                    self.violations.append(f"{rel}:{node.lineno}")
        self.generic_visit(node)

    def _is_allowed_location(self) -> bool:
        rel = self.file_path.relative_to(PROJECT_ROOT).as_posix()
        if rel == "sce/engine.py":
            return True
        if rel.startswith("tests/"):
            return True
        if rel == "scripts/run.py" and "_run_sce_enrichment" in self.function_stack:
            return True
        return False


def test_no_engine_fit_transform_outside_approved_scope():
    violations: list[str] = []

    for scan_dir in SCAN_DIRS:
        for py_file in scan_dir.rglob("*.py"):
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
            visitor = FitTransformCallVisitor(py_file)
            visitor.visit(tree)
            violations.extend(visitor.violations)

    assert not violations, "Unsafe engine.fit_transform usage found:\n" + "\n".join(violations)
