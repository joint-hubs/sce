"""
Tests for the autodocs package.

Tests cover:
- Parser (module headers, component decorators)
- Validator (circular deps, export verification)
- Generators (mermaid, markdown, injector)
"""

from pathlib import Path
from textwrap import dedent

import pytest
from autodocs.generators.injector import (
    find_markers,
    inject_content,
)
from autodocs.generators.markdown import (
    _anchor,
    _status_badge,
    generate_api_reference,
    generate_module_index,
)
from autodocs.generators.mermaid import (
    _node_id,
    generate_component_diagram,
    generate_dependency_graph,
)
from autodocs.models import ModuleMetadata, ProjectMetadata
from autodocs.parser import (
    parse_component_decorators,
    parse_module_header,
    scan_project,
)
from autodocs.validator import (
    MetadataValidator,
    Severity,
    find_circular_dependencies,
    get_actual_exports,
    validate_project,
)

# ============================================================================
# Parser Tests
# ============================================================================


class TestParseModuleHeader:
    """Tests for parse_module_header function."""

    def test_complete_header(self):
        source = dedent('''
        """
        @module: sce.engine
        @depends: sce.config, sce.stats
        @exports: StatisticalContextEngine
        @paper_ref: Algorithm 1
        @data_flow: raw -> enriched
        """
        
        class Engine:
            pass
        ''')

        meta = parse_module_header(source)

        assert meta is not None
        assert meta.module == "sce.engine"
        assert meta.depends == ["sce.config", "sce.stats"]
        assert meta.exports == ["StatisticalContextEngine"]
        assert meta.paper_ref == "Algorithm 1"
        assert meta.data_flow == "raw -> enriched"

    def test_minimal_header(self):
        source = dedent('''
        """
        @module: sce.cli
        @depends: 
        @exports: main
        """
        ''')

        meta = parse_module_header(source)

        assert meta is not None
        assert meta.module == "sce.cli"
        assert meta.depends == []  # Empty string parses to empty list
        assert meta.exports == ["main"]

    def test_missing_header_returns_none(self):
        source = "def foo(): pass"
        assert parse_module_header(source) is None

    def test_no_module_field_returns_none(self):
        source = dedent('''
        """
        @depends: something
        @exports: Foo
        """
        ''')
        assert parse_module_header(source) is None

    def test_single_quoted_docstring(self):
        source = dedent("""
        '''
        @module: sce.test
        @depends: 
        @exports: Test
        '''
        """)

        meta = parse_module_header(source)
        assert meta is not None
        assert meta.module == "sce.test"

    def test_status_field(self):
        source = dedent('''
        """
        @module: sce.experimental
        @depends: 
        @exports: Exp
        @status: EXPERIMENTAL - Test coverage 17%
        """
        ''')

        meta = parse_module_header(source)

        assert meta is not None
        assert meta.is_experimental is True
        assert meta.is_todo is False


class TestParseComponentDecorators:
    """Tests for parse_component_decorators function."""

    def test_component_decorator(self):
        source = dedent("""
        from sce.meta import component
        
        @component(
            name="TestEngine",
            responsibility="Does testing",
            depends_on=["Config"]
        )
        class TestEngine:
            pass
        """)

        components = parse_component_decorators(source, Path("test.py"))

        assert len(components) == 1
        assert components[0].name == "TestEngine"
        assert components[0].responsibility == "Does testing"
        assert components[0].depends_on == ["Config"]
        assert components[0].class_name == "TestEngine"

    def test_multiple_components(self):
        source = dedent("""
        @component(name="A", responsibility="First")
        class A:
            pass
        
        @component(name="B", responsibility="Second")
        class B:
            pass
        """)

        components = parse_component_decorators(source, Path("test.py"))

        assert len(components) == 2
        names = {c.name for c in components}
        assert names == {"A", "B"}

    def test_no_decorators(self):
        source = dedent("""
        class Plain:
            pass
        """)

        components = parse_component_decorators(source, Path("test.py"))
        assert len(components) == 0

    def test_syntax_error_returns_empty(self):
        source = "this is not valid python {"
        components = parse_component_decorators(source, Path("test.py"))
        assert components == []


# ============================================================================
# Validator Tests
# ============================================================================


class TestCircularDependencies:
    """Tests for circular dependency detection."""

    def test_simple_cycle(self):
        graph = {
            "a": ["b"],
            "b": ["c"],
            "c": ["a"],
        }

        cycles = find_circular_dependencies(graph)

        assert len(cycles) >= 1
        # Cycle should contain all three nodes
        cycle_nodes = set(cycles[0][:-1])  # Exclude repeated last element
        assert cycle_nodes == {"a", "b", "c"}

    def test_no_cycle(self):
        graph = {
            "a": ["b"],
            "b": ["c"],
            "c": [],
        }

        cycles = find_circular_dependencies(graph)
        assert len(cycles) == 0

    def test_self_loop(self):
        graph = {
            "a": ["a"],
        }

        cycles = find_circular_dependencies(graph)
        assert len(cycles) >= 1

    def test_empty_graph(self):
        graph = {}
        cycles = find_circular_dependencies(graph)
        assert len(cycles) == 0


class TestGetActualExports:
    """Tests for get_actual_exports function."""

    def test_extracts_classes(self):
        source = dedent("""
        class PublicClass:
            pass
        
        class _PrivateClass:
            pass
        """)

        exports = get_actual_exports(source)

        assert "PublicClass" in exports
        assert "_PrivateClass" not in exports

    def test_extracts_functions(self):
        source = dedent("""
        def public_func():
            pass
        
        def _private_func():
            pass
        """)

        exports = get_actual_exports(source)

        assert "public_func" in exports
        assert "_private_func" not in exports

    def test_respects_all(self):
        source = dedent("""
        __all__ = ["Exported", "Also"]
        
        class Exported:
            pass
        
        class NotExported:
            pass
        """)

        exports = get_actual_exports(source)

        assert exports == {"Exported", "Also"}


class TestMetadataValidator:
    """Tests for MetadataValidator class."""

    def test_validates_empty_exports(self):
        project = ProjectMetadata(
            modules=[ModuleMetadata(module="sce.empty", depends=[], exports=[])]
        )

        validator = MetadataValidator(project)
        results = validator._validate_module(project.modules[0])

        warnings = [r for r in results if r.severity == Severity.WARNING]
        assert any("MISSING_EXPORTS" in w.rule for w in warnings)

    def test_validates_external_dependency(self):
        project = ProjectMetadata(
            modules=[
                ModuleMetadata(
                    module="sce.engine", depends=["scipy", "sklearn"], exports=["Engine"]
                )
            ]
        )

        validator = MetadataValidator(project)
        results = validator._validate_dependencies()

        # External deps are INFO, not ERROR
        infos = [r for r in results if r.severity == Severity.INFO]
        assert len(infos) >= 2


class TestValidateProject:
    """Tests for validate_project function."""

    def test_success_on_valid_project(self):
        project = ProjectMetadata(
            modules=[
                ModuleMetadata(module="sce.a", depends=[], exports=["A"]),
                ModuleMetadata(module="sce.b", depends=["sce.a"], exports=["B"]),
            ]
        )

        success, report = validate_project(project)

        assert success is True
        assert "passed" in report.lower() or len(report) > 0


# ============================================================================
# Mermaid Generator Tests
# ============================================================================


class TestMermaidGenerator:
    """Tests for Mermaid diagram generation."""

    def test_node_id_conversion(self):
        assert _node_id("sce.engine") == "sce_engine"
        assert _node_id("sce.io") == "sce_io"
        assert _node_id("my-package") == "my_package"

    def test_dependency_graph_deterministic(self):
        """Same input should produce identical output."""
        project = ProjectMetadata(
            modules=[
                ModuleMetadata(module="b", depends=["a"]),
                ModuleMetadata(module="a", depends=[]),
            ]
        )

        output1 = generate_dependency_graph(project)
        output2 = generate_dependency_graph(project)

        assert output1 == output2

    def test_dependency_graph_contains_nodes(self):
        project = ProjectMetadata(
            modules=[
                ModuleMetadata(module="sce.engine", depends=["sce.config"]),
                ModuleMetadata(module="sce.config", depends=[]),
            ]
        )

        output = generate_dependency_graph(project)

        assert "sce_engine" in output
        assert "sce_config" in output
        assert "sce_engine --> sce_config" in output

    def test_component_diagram_empty(self):
        project = ProjectMetadata()
        output = generate_component_diagram(project)

        assert "C4Component" in output


# ============================================================================
# Markdown Generator Tests
# ============================================================================


class TestMarkdownGenerator:
    """Tests for Markdown documentation generation."""

    def test_anchor_generation(self):
        assert _anchor("sce.engine") == "sceengine"
        assert _anchor("sce.io") == "sceio"

    def test_status_badge_stable(self):
        module = ModuleMetadata(module="test", status=None)
        badge = _status_badge(module)
        assert "Stable" in badge

    def test_status_badge_experimental(self):
        module = ModuleMetadata(module="test", status="EXPERIMENTAL")
        badge = _status_badge(module)
        assert "Experimental" in badge

    def test_api_reference_has_toc(self):
        project = ProjectMetadata(
            modules=[
                ModuleMetadata(module="sce.engine", exports=["Engine"]),
                ModuleMetadata(module="sce.config", exports=["Config"]),
            ]
        )

        output = generate_api_reference(project)

        assert "## Table of Contents" in output
        assert "[sce.engine]" in output
        assert "[sce.config]" in output

    def test_module_index_compact(self):
        project = ProjectMetadata(
            modules=[
                ModuleMetadata(
                    module="sce.engine", exports=["A", "B", "C", "D"], paper_ref="Algo 1"
                ),
            ]
        )

        output = generate_module_index(project)

        assert "`sce.engine`" in output
        assert "+2" in output  # 4 exports, show 2, "+2"
        assert "Algo 1" in output


# ============================================================================
# Injector Tests
# ============================================================================


class TestInjector:
    """Tests for README injection."""

    def test_inject_content_simple(self):
        readme = dedent("""
        # Title
        
        <!-- AUTODOC:TEST:START -->
        old content
        <!-- AUTODOC:TEST:END -->
        
        ## Footer
        """)

        result = inject_content(readme, "TEST", "new content")

        assert "new content" in result
        assert "old content" not in result
        assert "# Title" in result
        assert "## Footer" in result

    def test_inject_preserves_structure(self):
        readme = dedent("""
        Before
        
        <!-- AUTODOC:A:START -->
        <!-- AUTODOC:A:END -->
        
        Middle
        
        <!-- AUTODOC:B:START -->
        <!-- AUTODOC:B:END -->
        
        After
        """)

        result = inject_content(readme, "A", "content A")
        result = inject_content(result, "B", "content B")

        assert "Before" in result
        assert "Middle" in result
        assert "After" in result
        assert "content A" in result
        assert "content B" in result

    def test_find_markers(self):
        readme = dedent("""
        <!-- AUTODOC:GRAPH:START -->
        <!-- AUTODOC:GRAPH:END -->
        
        <!-- AUTODOC:INDEX:START -->
        <!-- AUTODOC:INDEX:END -->
        """)

        markers = find_markers(readme)

        assert set(markers) == {"GRAPH", "INDEX"}

    def test_no_markers_no_change(self):
        readme = "# No markers here"
        result = inject_content(readme, "TEST", "content")
        assert result == readme


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for full workflow."""

    def test_scan_actual_project(self):
        """Test scanning the actual sce package."""
        sce_path = Path(__file__).parent.parent / "sce"

        if not sce_path.exists():
            pytest.skip("sce package not found")

        project = scan_project(sce_path)

        # Should find multiple modules
        assert len(project.modules) > 0

        # Should find engine module
        engine = project.get_module("sce.engine")
        assert engine is not None

        # Should have components
        assert len(project.components) >= 1
