# Migration Checklist (LangGraph Examples)

Use this checklist during edits:

- Imports reflect current APIs used in the repo examples.
- Graph includes `START` and (when appropriate) explicit `END` edges.
- State typing is coherent between parent graph and subgraph.
- Subgraph invocation passes required keys for subgraph state.
- Message-based flows preserve role semantics (`HumanMessage`, `AIMessage`, `SystemMessage`).
- Demo section remains executable and easy to read.
- Run `python -m py_compile` on edited Python files.
