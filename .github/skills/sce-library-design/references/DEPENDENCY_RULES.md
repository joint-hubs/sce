# Dependency Rules

## Allowed Imports

- `sce.engine` can import: `sce.stats`, `sce.config`, `sce.meta`
- `sce.stats` can import: standard libs, numpy/pandas
- `sce.config` can import: standard libs, pydantic/dataclasses
- `sce.pipeline` can import: `sce.engine`, `sce.config`, `sce.io`
- `sce.io` can import: standard libs, pandas, `sce.config`
- `sce.meta` can import: standard libs only

## Forbidden Imports

- `sce.io` must not import `sce.engine`
- `sce.stats` must not import `sce.pipeline`
- Scripts should not be imported by library modules

## Header Metadata

Every module must include a header block (see template).
CI should fail if missing or invalid.
