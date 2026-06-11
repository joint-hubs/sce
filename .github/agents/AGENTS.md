# Agent System for SCE Project

**Repository:** [github.com/joint-hubs/sce](https://github.com/joint-hubs/sce)

## Philosophy

Agents are **specialized roles** defined as `.agent.md` files that load specific tools and follow focused procedures. They complement the general Copilot by providing deep expertise in narrow domains.

Custom agents are stored in `.github/agents/` and appear in the VS Code Chat agents dropdown.

---

## Available Agents

### auditor.agent.md — Paper Accuracy Agent

**Purpose**: Verify code matches paper equations exactly.

**Tools**: `search`, `codebase`, `usages`

**Use When**:
- "audit the engine"
- "verify against paper"
- "check for leakage"
- "does this match Algorithm 1?"

---

### experimenter.agent.md — Reproducibility Agent

**Purpose**: Run experiments, validate results, ensure reproducibility.

**Tools**: `terminal`, `search`, `codebase`

**Use When**:
- "run all experiments"
- "reproduce paper results"
- "validate dataset"
- "check reproducibility"

---

## Deferred Agents

The following agents are planned but not yet implemented:

| Agent | Purpose | Status |
|---|---|---|
| `publisher` | Generate paper figures/tables | ⏸️ Deferred |
| `architect` | Enforce architecture standards | ⏸️ Deferred |
| `shipper` | Prepare release | ⏸️ Deferred |

---

## Agent Activation

Agents are activated by:
1. Selecting from the VS Code Chat agents dropdown
2. Using `@agent-name` syntax in chat
3. Keyword matching in user query

When activated, an agent:
1. Loads its tools and instructions
2. Follows its defined procedure
3. Documents findings in skills/references

---

## Creating New Agents

Create a new `.agent.md` file in `.github/agents/` with this structure:

```markdown
---
name: Agent Name
description: What this agent does and when to use it.
tools: ['tool1', 'tool2']
---

# Agent Instructions

Your detailed instructions here...
```

See [VS Code Custom Agents docs](https://code.visualstudio.com/docs/copilot/customization/custom-agents) for full specification.
