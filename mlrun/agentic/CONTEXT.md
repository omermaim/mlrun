# Agentic Module — Controller Removal & Memory Gateway

> **Re-entry instructions**: Read this file for full context on the `agentic-remove-controller` branch.

## Branch & Commits

**Branch**: `agentic-remove-controller` (based on `agentic-ai-declerative`)

6 commits:

1. `270044a46` — [Agentic] Replace controller-backed SessionStore with MLRun artifact storage
2. `a2957eef4` — [Agentic] Code cleanup: structured logging, configurable chains, DRY strategy graph
3. `0f9ec8c4b` — [Agentic] Remove controller dependency: delete ControllerClient, config class, schema cleanup
4. `84407a160` — [Agentic] Add agentstores memory gateway: STM sessions + LTM
5. `6d802926f` — [Agentic] Added memory_saver memory_loader
6. `dbc1f9e86` — [Agentic] Fix ChainRunner._do to always propagate events downstream

Plus two follow-up fixes:
- `3db335e4c` — [Agentic] Add agentstores memory gateway: STM sessions + LTM facts
- `fcce5a451` — removed unused code in refine.py

---

## What Was Done

1. **Deleted `controller_client.py`** — entire HTTP client for genai-factory controller (~205 lines)
2. **Deleted `WorkflowServerConfig` class** — ~50-line Pydantic config model in `config.py`; chains no longer need a shared config object
3. **Rewritten `sessions.py`** — pluggable `SessionStore` interface, now backed by `agentstores` (`SessionStore(session_store=ASSessionStore)`)
4. **New chains: `MemoryLoader` and `MemorySaver`** — LTM fact retrieval and extraction via `agentstores`
5. **Memory-aware updates** — `RefineQuery` gets `MEMORY_AWARE_REFINER_PROMPT`; `Communicator` prepends facts to query
6. **All chains self-sufficient** — accept `model_name`/`temperature` in constructor, create LLM in `post_init()`
7. **DRY strategy compilation** — extracted `compile_strategy_graph()` in `strategies.py`, shared by `DeclarativeTeam` and `DeclarativeTeamRouter`
8. **Schema cleanup** — deleted `APIResponse`, `APIDictResponse`, `OutputMode`; fixed `session_id` → `session_name` bug; `project_id` now optional
9. **Structured logging** — all `print()` and f-string logs replaced with `mlrun.utils.logger` + kwargs
10. **Cleaned up `WorkflowEvent`** — removed dead fields (`workflow_id`, `user`, `db_session`); added `memory_context: list[dict] = []` as proper field; added `__str__`/`__repr__`
11. **Removed `@property llm` trap** — all chains create LLMs in `post_init()` to avoid `deepcopy`/`to_dict()` issues with thread locks
12. **`SessionLoader` → `ChainRunner` subclass** — fixes sync engine handler issue; has `_do()`, `__call__()`, `_run()`
13. **Auto-wiring `graph_initializer`** — new `mlrun/agentic/serving.py` reads env vars (`AGENTIC_DB_PATH`, `AGENTIC_GATEWAY_URL`, etc.)

---

## Bugs Encountered

### Bug 1: `ChainRunner._do()` — Empty Dict Stops the Chain (CRITICAL)

**Symptom**: Serving graph hangs when `MemoryLoader`/`MemorySaver` is in the chain.
**Root cause**: `_do_downstream()` was inside `if resp:` — empty dict `{}` is falsy, so steps returning `{}` silently stopped the chain.
**Fix**: Moved `_do_downstream()` outside the `if resp:` block.
**Note**: Same bug exists in genai-factory's `ChainRunner._do()`.

### Bug 2: `deepcopy` Fails with Class Instances in `.to()`

**Symptom**: `TypeError: cannot pickle '_thread.RLock' object` when building graph with instances.
**Root cause**: `params_to_step()` calls `deepcopy(instance.to_dict())` — pydantic v2 models have internal `_thread.RLock`.
**Fix**: Use string class paths instead of instances for `.to()`, or create LLMs in `post_init()` (not `__init__`).

### Bug 3: `agentstores` Missing Route Module

**Symptom**: `ModuleNotFoundError: No module named 'agentstores.backend.routes.auth'`.
**Root cause**: Eager import of full server in `agentstores/backend/__init__.py`.
**Fix**: Fixed in agentstores repo (branch `fix/missing-auth-routes`).

### Bug 4: FTS5 Syntax Error with Special Characters

**Symptom**: `sqlite3.OperationalError: fts5: syntax error near "!"`.
**Root cause**: User queries passed directly to FTS5 without escaping.
**Fix**: `Client(db_path=..., enable_fts=False)` — disabled by default in `serving.py`.

### Bug 5: `SessionLoader` Has No Handler in Sync Engine

**Symptom**: `MLRunInvalidArgumentError: step session-loader does not have a handler`.
**Root cause**: `SessionLoader` inherited from `storey.Flow` (no `__call__`), not `ChainRunner`.
**Fix**: Made `SessionLoader` a `ChainRunner` subclass (item #12 above).

### Bug 6: `WorkflowEvent.to_dict()` Fails with Agentstores Session

**Symptom**: `AttributeError` serializing event with agentstores `Session` handle.
**Fix**: Duck-typing check: `hasattr(self.session, "to_dict")` with fallback.

### Bug 7: Mock Server Returns Raw WorkflowEvent Object

**Symptom**: `mock_server.test()` returns `<WorkflowEvent object at 0x...>`.
**Workaround**: Access answer via `resp.results.get("answer")`.

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **No config class** — `config.py` has only stateless factory functions (`get_llm`, `get_vector_db`, `get_embedding_function`) taking plain dicts | Eliminates coupling; chains are testable standalone |
| **Pluggable SessionStore** — interface is `read_state(event)` + `save(event)` | Allows swapping backends without changing graph wiring |
| **Self-sufficient chains** — LLM params in constructor, created in `post_init()` | No shared config injection needed; avoids `@property` deepcopy trap |
| **DRY strategy compilation** — `compile_strategy_graph()` in `strategies.py` | Single source of truth for team + router strategy dispatch |
| **Optional agentstores** — try/except imports in `__init__.py` | Graceful degradation if agentstores not installed |
| **Memory context as field** — `WorkflowEvent.memory_context: list[dict] = []` | Type-safe access, no state-dict indirection |

---

## Files Modified

| File | Action | Description |
|------|--------|-------------|
| `controller_client.py` | Deleted | Entire controller HTTP client |
| `config.py` | Rewritten | Deleted config class, kept factory functions |
| `sessions.py` | Rewritten | Agentstores-backed SessionStore |
| `serving.py` | New | Default graph_initializer with env-var config |
| `chains/base.py` | Modified | Logging + fixed `_do()` empty dict bug |
| `chains/memory_loader.py` | New | LTM fact retrieval |
| `chains/memory_saver.py` | New | LTM fact extraction & storage |
| `chains/refine.py` | Modified | Configurable params + memory-aware prompt |
| `chains/communicator.py` | Modified | Configurable params + memory context |
| `chains/intent_classifier.py` | Modified | Configurable params |
| `chains/hallucination_guardrail.py` | Modified | Configurable params |
| `chains/retrieval.py` | Modified | Configurable params, `from_config` → `from_dicts` |
| `chains/declarative/strategies.py` | Modified | Extracted `compile_strategy_graph()` |
| `chains/declarative/team.py` | Modified | Uses shared strategy compilation |
| `chains/declarative/router.py` | Modified | Uses shared strategy compilation |
| `chains/declarative/runner.py` | Modified | Removed dead `_is_likely_path()` |
| `schemas/workflow.py` | Modified | Optional project_id, agentstores Session serialization |
| `schemas/base.py` | Modified | Removed controller-only classes |
| `schemas/session.py` | Modified | Removed duplicate `to_dict()` |
| `schemas/__init__.py` | Modified | Cleaned exports |
| `utils.py` | Modified | Uses `mlrun.utils.logger` instead of stdlib |
| `__init__.py` | Modified | Removed ControllerClient + config class exports, try/except for agentstores |
| `chains/__init__.py` | Modified | Updated exports (`get_retriever_from_dicts`), try/except for memory chains |
| `tests/experimental/test_session_store.py` | New | 21 unit tests |

---

## Remaining Work

### 1. Agentstores FTS Escaping (Low priority)

SQLite FTS5 syntax errors on special characters. Current workaround: `AGENTIC_ENABLE_FTS=0`. Fix belongs in the agentstores library.

### 2. Memory Integration for Declarative Agents (Medium priority)

`AgentsAtScaleDeployer` doesn't wire `MemoryLoader`/`MemorySaver`. Options:
- Add `with_memory=True` param to `AgentsAtScaleDeployer.build()`
- Make `DeclarativeAgent` itself memory-aware

### 3. Convert `WorkflowEvent` to Dataclass (Low priority)

Deferred — current plain class works well. Blocker: need to check if callers rely on `**kwargs` catch-all constructor.

### 4. Deployer `.to(instance)` Pattern (Low priority — informational)

Deployer passes class instances to `.to()`. No issue currently (classes store only serializable data), but new chain classes with non-serializable `__init__` params should use string class paths.

### 5. Tool Registry (from agentstores Phase 3)

Planned agentstores feature for tool/function registration. Not started.

---

## Key Paths

| Path | Description |
|------|-------------|
| `mlrun/agentic/` | Agentic AI module (chains, declarative agents/teams, schemas) |
| `mlrun/agentic/serving.py` | Default graph_initializer with env-var config |
| `mlrun/agentic/examples/a2aclient/` | Programmatic chain graph example |
| `tests/experimental/test_session_store.py` | 21 tests for SessionStore + MemoryLoader + MemorySaver |
| `/Users/Omer_Mimon/PycharmProjects/agentstores/` | agentstores library (session STM + LTM client) |
| `/Users/Omer_Mimon/PycharmProjects/genai-factory/` | Source project (genai-factory) |
