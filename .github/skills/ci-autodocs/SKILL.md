---
name: ci-autodocs
description: CI pipeline for auto-generating documentation and architecture diagrams. Use when setting up CI, generating diagrams, updating README automatically, or validating metadata. Triggers include CI, auto-generate, diagram, Mermaid, dependency graph, architecture.
---

# CI Auto-Documentation

## Quick Reference

| Document | Purpose |
|---|---|
| references/METADATA_SPEC.md | Module header specification |
| docs/autogen/autodocs_design.md | System design document |
| docs/autogen/tasks/ | Implementation task files |

## Implementation Status

| Task | Status | Description |
|------|--------|-------------|
| Task 01 | 🔲 TODO | Parser models and core implementation |
| Task 02 | 🔲 TODO | Metadata validator |
| Task 03 | 🔲 TODO | Mermaid diagram generator |
| Task 04 | 🔲 TODO | Markdown API docs generator |
| Task 05 | 🔲 TODO | README injection system |
| Task 06 | 🔲 TODO | CI/CD integration |

## When to Use This Skill

- You need to parse code metadata for diagrams
- You need to inject results into README
- You need to validate dependency rules in CI
- You're implementing any autodocs component

## Procedure

1. Parse module headers from source files.
2. Build dependency and component graphs.
3. Generate Mermaid diagrams.
4. Inject artifacts into README markers.

## Gotchas

- Fail CI on missing metadata headers.
- Keep Mermaid generation deterministic (sort all nodes/edges).
- Use `<!-- AUTODOC:NAME:START -->` markers for injection.
- AST parsing required for `@component` decorators.

## Key Files

```
sce/
├── autodocs/              # Autodocs package (to implement)
│   ├── models.py          # Data classes
│   ├── parser.py          # Parsing logic
│   ├── validator.py       # Validation rules
│   ├── generators/
│   │   ├── mermaid.py     # Diagram generation
│   │   ├── markdown.py    # API docs
│   │   └── injector.py    # README injection
│   └── cli.py             # CLI interface
scripts/ci/
├── generate_docs.py       # CI script (wraps autodocs)
└── validate_metadata.py
docs/generated/            # Output directory
```
