# Module Responsibility Matrix

| Module | Responsibility | Notes |
|---|---|---|
| `sce.engine` | Core SCE algorithm and feature construction | No IO | 
| `sce.stats` | Aggregation functions and helpers | Pure functions | 
| `sce.config` | Config schemas and validation | No compute | 
| `sce.pipeline` | Orchestration of data → features → model | Thin wrappers | 
| `sce.io` | Load/save datasets and artifacts | No model logic | 
| `sce.meta` | Decorators for architecture tracking | Minimal deps | 
| `scripts/` | CLI entry points and automation | Keep <300 LOC | 

## File Size Rule
- Each module must stay under 300 LOC.
- Split by responsibility when growing.
