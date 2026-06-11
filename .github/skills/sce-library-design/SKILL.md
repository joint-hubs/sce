---
name: sce-library-design
description: SCE Python library architecture and coding standards. Use when creating new modules, refactoring code, checking dependencies, defining module responsibilities, or writing module headers. Triggers include module, architecture, dependency, imports, engine.py, pipeline.py, meta.py, component.
---

# SCE Library Design Standards

## Quick Reference

| Document | Purpose |
|---|---|
| references/MODULE_SPEC.md | Module responsibilities |
| references/DEPENDENCY_RULES.md | Import boundaries |
| docs/standards/code_metadata.md | Metadata rules |
| templates/module_header.txt | Required header template |

## When to Use This Skill

- Creating or refactoring modules
- Enforcing <300 LOC per file
- Adding new components or pipelines

## Procedure

1. Check MODULE_SPEC for where the change belongs.
2. Validate imports against DEPENDENCY_RULES.
3. Add module header metadata using the template.
4. Keep file under 300 LOC (split if needed).

## Common Patterns

- Core engine lives in a single file; helper stats in a separate module.
- IO and config are separated from computation.
- Pipelines should be composable and testable.

## Resources

- [Module responsibilities](references/MODULE_SPEC.md)
- [Dependency rules](references/DEPENDENCY_RULES.md)

## Gotchas

- Do not let `io` import `engine` (keep data layer thin).
- Avoid circular imports by using lightweight interfaces.
- Keep module header metadata aligned with real imports and exports; stale `@depends` or `@exports` values mislead graphify and autodoc workflows.
