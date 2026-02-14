---
name: langgraph-example-migrator
description: Migrate this repository's LangGraph/LangChain chapter examples to current APIs, preserving educational intent and Korean comments. Use when asked to modernize files (e.g., ch07 examples), update typing/state patterns, and keep runnable sample flow clear.
---

# LangGraph Example Migrator

Use this skill when a user asks to update tutorial/example code to newer LangGraph/LangChain APIs.

## Workflow

1. Locate target example file(s) and the matching JavaScript/Python counterpart if present.
2. Prefer minimal diffs that preserve learning intent, comments, and sample output flow.
3. Apply current patterns:
   - Explicit graph termination (`END`) when flow clarity benefits examples.
   - Built-in state helpers (e.g., `MessagesState`) where replacing custom boilerplate is beneficial.
   - Clear return type hints for node functions.
4. Keep adapter logic explicit for parent/subgraph state mapping.
5. Validate with lightweight syntax checks (`python -m py_compile <file>`).
6. Summarize *what changed* and *why it is newer API style*.

## Repository-specific conventions

- Keep Korean comments intact unless clearly incorrect.
- Avoid unnecessary refactors outside the target educational snippet.
- Prefer concise, readable code over framework-heavy abstractions.

## References

- Migration checklist: `references/migration-checklist.md`
