# Agent Skills Guide

> A comprehensive guide to creating, using, and understanding Agent Skills for AI-powered workflows.
ALWAYS SEARCH WEB USING LINKS REFERENCED HERE:
---

- https://code.visualstudio.com/docs/copilot/customization/overview
- https://code.visualstudio.com/docs/copilot/customization/agent-skills
- https://code.visualstudio.com/docs/copilot/customization/custom-agents
- https://code.visualstudio.com/docs/copilot/customization/prompt-files
- https://code.visualstudio.com/docs/copilot/customization/custom-instructions

## What Are Agent Skills?

Agent Skills are **folders of instructions, scripts, and resources** that AI agents can discover and use to perform specialized tasks. They follow an [open standard](https://agentskills.io/) that works across multiple AI tools.

At its core, a skill is a folder containing a `SKILL.md` file with:
- **Metadata** (`name` and `description`) — helps agents know when to use it
- **Instructions** — detailed guidance on how to perform the task
- **Optional resources** — scripts, examples, templates

```
my-skill/
├── SKILL.md          # Required: metadata + instructions
├── scripts/          # Optional: executable code
├── references/       # Optional: additional documentation
└── assets/           # Optional: templates, resources
```

---

## Why Skills Matter

### The Problem They Solve

Agents are increasingly capable, but they often lack the **context** they need to do real work reliably:

| Without Skills | With Skills |
|----------------|-------------|
| Agent guesses how to do task | Agent follows documented procedure |
| Repeating context every conversation | Context loaded automatically when relevant |
| Generic responses | Domain-specific expertise |
| One-size-fits-all | Specialized workflows |

## Skills vs Custom Instructions

| Aspect | Skills | Custom Instructions |
|--------|--------|---------------------|
| **Purpose** | Specialized capabilities and workflows | Coding standards and guidelines |
| **Portability** | Works across VS Code, CLI, and coding agents | VS Code and GitHub.com only |
| **Content** | Instructions, scripts, examples, resources | Instructions only |
| **Scope** | Task-specific, loaded on-demand | Always applied (or via glob patterns) |
| **Standard** | Open standard (agentskills.io) | VS Code-specific |

**Use Skills when you want to:**
- Create reusable capabilities that work across different AI tools
- Include scripts, examples, or other resources alongside instructions
- Share capabilities with the wider AI community
- Define specialized workflows (testing, debugging, deployment, etc.)

**Use Custom Instructions when you want to:**
- Define project-specific coding standards
- Set language or framework conventions
- Specify code review or commit message guidelines
- Apply rules based on file types using glob patterns

---

### How It Works

1. **`copilot-instructions.md`** — Loaded for EVERY agent automatically
2. **`.instructions.md` files** — Loaded based on `applyTo` glob patterns
3. **`AGENTS.md`** (optional) — Cross-agent instructions

### smart agent Structure

```
.github/
├── copilot-instructions.md           # Global: architecture, patterns, paths
├── GUIDE.md                          # This file
├── instructions/
│   ├── contributor.instructions.md   # All contributors: delivery workflow
│   └── orchestrator.instructions.md  # Orchestrator only: sprint hosting
├── agents/                           # Agent personality files
│   ├── orchestrator.agent.md
│   ├── tech_lead.agent.md
│   └── ...
└── skills/                           # Domain expertise
    ├── developer-tasks/
    ├── sprint-delivery/
    └── ...
```



## When to Create a New Skill

### ✅ Create a Skill When:

| Scenario | Example |
|----------|---------|
| **Recurring workflow** | Weekly review process, daily journal filling |
| **Domain expertise** | Understanding your vault structure, time tracking system |
| **Procedural knowledge** | Step-by-step instructions for complex tasks |
| **Shared context** | Information multiple agents need |
| **External tool usage** | Scripts, templates, or automation helpers |

### ❌ Don't Create a Skill When:

| Scenario | Better Alternative |
|----------|-------------------|
| One-time task | Just ask the agent directly |
| Project coding standards | Use custom instructions (`.github/instructions/`) |
| Simple preferences | Use VS Code settings or prompts |
| Agent personality | Define in agent file (`.github/agents/`) |

### Decision Tree

```
Is this knowledge/procedure needed repeatedly?
├─ No → Just ask the agent directly
└─ Yes → Is it coding standards or guidelines?
    ├─ Yes → Use custom instructions
    └─ No → Does it involve procedures, tools, or domain knowledge?
        ├─ Yes → CREATE A SKILL ✅
        └─ No → Consider if it belongs in agent personality
```

---

## How Progressive Disclosure Works

Skills use a three-level loading system for efficiency:

### Level 1: Discovery (~100 tokens)
At startup, agents load only `name` and `description` from each skill's frontmatter. This is enough to know when a skill might be relevant.

### Level 2: Instructions (<2000 tokens recommended)
When a task matches a skill's description, the agent loads the full `SKILL.md` body into context.

### Level 3: Resources (as needed)
Additional files (scripts, references, assets) are loaded only when the agent references them.

**This means you can have many skills without consuming context** — only what's relevant loads.

---

## Creating a Skill

### Step 1: Create the Folder Structure

```
.github/skills/
└── my-skill/
    └── SKILL.md
```

### Step 2: Write the SKILL.md File

```markdown
---
name: my-skill
description: |
  Brief description of what this skill does and when to use it.
  Be specific about trigger phrases and use cases.
  Maximum 1024 characters.
metadata:
  author: your-name
  version: "1.0"
---

# Skill Title

## Purpose
What problem does this skill solve?

## When to Use
- Trigger phrase 1
- Trigger phrase 2
- Specific scenarios

## Procedure
Step-by-step instructions...

## Examples
Input/output examples...

## Tools Used
Which MCP tools or resources this skill uses...
```

### Step 3: Add Optional Resources

```
my-skill/
├── SKILL.md
├── scripts/
│   └── automation.py      # Executable scripts
├── references/
│   └── REFERENCE.md       # Detailed documentation
└── assets/
    └── template.md        # Templates
```

---

## SKILL.md Specification

### Required Frontmatter

| Field | Required | Rules |
|-------|----------|-------|
| `name` | Yes | Max 64 chars, lowercase, hyphens only, no consecutive hyphens, must match folder name |
| `description` | Yes | Max 1024 chars, describes what skill does AND when to use it |

### Optional Frontmatter

| Field | Description |
|-------|-------------|
| `license` | License name or reference to bundled file |
| `compatibility` | Environment requirements (max 500 chars) |
| `metadata` | Key-value pairs for additional info (author, version) |
| `allowed-tools` | Space-delimited list of pre-approved tools (experimental) |

### Body Content

No format restrictions. Write whatever helps agents perform the task effectively.

**Recommended sections:**
- Purpose/overview
- When to use (trigger phrases)
- Step-by-step procedures
- Examples of inputs and outputs
- Common edge cases
- Tools/resources used



## Best Practices

### 1. Keep SKILL.md Under 500 Lines
Move detailed reference material to separate files in `references/`.

### 2. Be Specific About Triggers
Include exact phrases users might say that should activate this skill.

### 3. Use Relative Paths for Resources
```markdown
See [the reference guide](references/REFERENCE.md) for details.
Run the extraction script: `scripts/extract.py`
```

### 4. Document Tools Used
If your skill uses specific MCP tools, list them clearly.

### 5. Include Examples
Show expected inputs and outputs so agents understand the task.

### 6. Test Your Skill
Verify that agents correctly activate and follow your skill's instructions.


## Quick Reference

### Minimum Viable Skill

```markdown
---
name: my-skill
description: What it does. Use when user says X, Y, or Z.
---

# My Skill

Instructions for the agent...
```

### Folder Location

```
.github/skills/       # Project skills (recommended)
~/.github/skills/     # Personal skills (user profile)
```

### Naming Rules

- Lowercase only
- Hyphens for spaces
- No consecutive hyphens
- 1-64 characters
- Must match folder name


