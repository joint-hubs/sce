# Metadata Specification

## Required Header Fields

- `@module`: Fully-qualified module path
- `@depends`: Comma-separated internal deps
- `@exports`: Public symbols

## Optional Fields

- `@paper_ref`: Algorithm/equation references
- `@data_flow`: Compact data flow description

## Parser Rules

- First docstring in module must contain the metadata block.
- Unknown fields are allowed but ignored.
- Lines must use `key: value` format.
