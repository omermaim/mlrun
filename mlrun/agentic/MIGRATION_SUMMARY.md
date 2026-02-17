# Agentic Module: Controller Removal Migration

## Overview

This document summarizes the changes made to decouple `mlrun/agentic` from the
genai-factory controller dependency. The work is organized into three logical
areas, all on the `agentic-remove-controller` branch (based on
`agentic-ai-declerative`).

**Scope:** 20 files changed, +335 / -515 lines (net -180 lines)

---

## 1. SessionStore — Artifact-Backed via MLRun Project

### Problem

`SessionStore` was a thin wrapper around `ControllerClient`. Every operation
(`read_state`, `save`) made HTTP calls to a live genai-factory controller
endpoint (`get_user()`, `get_session()`, `update_session()`).

### Solution

Replaced the controller-backed session store with one backed by MLRun project
artifacts.

**File: `sessions.py`** (+78 / -18)

| Aspect | Before | After |
|--------|--------|-------|
| Constructor | `SessionStore(client)` | `SessionStore(project, default_username="guest")` |
| Read | `client.get_user()` + `client.get_session()` | `project.get_artifact("session-{name}")` |
| Save | `client.update_session()` | `project.log_artifact()` with JSON body + labels |
| User lookup | Required (`client.get_user()`) | Removed — username comes from event or defaults to `"guest"` |
| Error handling | Controller HTTP errors | Falls back to fresh session on any exception |

Artifact format:
- **Key:** `session-{session_name}`
- **Body:** JSON with `name`, `owner_id`, `history`, `workflow_id`
- **Labels:** `kind=chat-session`, `session_owner={username}`

### Design Decision

The `SessionStore` interface (`read_state(event)` + `save(event)`) is designed
as a pluggable interface. The current implementation uses MLRun artifacts, but a
future Memory Gateway backend will implement the same interface.

### Tests

**File: `tests/experimental/test_session_store.py`** (new, 148 lines, 9 tests)

| Test Class | Coverage |
|------------|----------|
| `TestSessionStoreInit` | Constructor, default username |
| `TestSessionStoreArtifactKey` | Key format (`session-{name}`) |
| `TestSessionStoreReadState` | Load existing session, create fresh on not-found, default username fallback, skip when no session_name |
| `TestSessionStoreSave` | `log_artifact()` call verification (key, body, labels), skip when no session_name |

All tests use `unittest.mock.MagicMock` for the MLRun project — no real cluster
needed.

---

## 2. Code Cleanup — Dead Code, DRY, Structured Logging, Configurable Chains

### 2a. Structured Logging

Replaced `print()` statements and f-string logging with `mlrun.utils.logger`
(MLRun's structured logger supporting keyword arguments).

**`utils.py`** (-3 / +2):
```python
# Before
import logging
logger = logging.getLogger("mlrun.agentic")
logger.addHandler(logging.StreamHandler())

# After
import mlrun.utils
logger = mlrun.utils.logger
```

**`chains/base.py`**: `print("step name: ", self.name)` became
`logger.debug("Running chain step: %s", self.name)`

**`chains/retrieval.py`** and **`chains/refine.py`**: All f-string log calls
converted to structured kwargs:
```python
# Before
logger.debug(f"Question: {event.query}\nChat history: {chat_history}")

# After
logger.debug("Refine query", question=event.query, chat_history=chat_history)
```

### 2b. Configurable Chains

All chains previously hardcoded their LLM model and temperature. Now each
accepts `model_name` and `temperature` in its constructor and creates the LLM
lazily via a `@property`.

**Pattern applied to all chains:**
```python
# Before
class SomeChain(ChainRunner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(model="gpt-4", temperature=0.5)  # hardcoded
        return self._llm

# After
class SomeChain(ChainRunner):
    def __init__(self, model_name="gpt-4", temperature=0.5, **kwargs):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._temperature = temperature
        self._llm = None

    @property
    def llm(self):
        if not self._llm:
            self._llm = ChatOpenAI(
                model=self._model_name, temperature=self._temperature
            )
        return self._llm
```

**Chains updated:**

| Chain | Default model | Default temperature |
|-------|--------------|-------------------|
| `RefineQuery` | gpt-4 | 0 |
| `Communicator` | gpt-4 | 0.5 |
| `IntentClassifier` | gpt-4 | 0.5 |
| `HallucinationGuardrail` | gpt-4o-mini | 0 |
| `MultiRetriever` | gpt-4 | 0 |

**`RefineQuery` additional changes:**
- Removed dependency on `get_llm(self.context._config)` in `post_init`
- Removed dead `_refine_prompt_template` (duplicate of
  `CONVERSATION_CONTEXT_REFINER_PROMPT` with worse wording)
- `get_refine_chain(config, ...)` became
  `get_refine_chain(model_name="gpt-4", temperature=0, ...)`

**`MultiRetriever` additional changes:**
- Now accepts `vector_store_args` and `embeddings_args` dicts in constructor
- Removed `self.llm = self.llm or get_llm(self.context._config)` from `post_init`
- Removed `self.default_collection = self.context._config.default_collection()`
- Added `_get_vector_db()` helper using factory functions directly

**`DocumentRetriever` rename:**
- `from_config(cls, config, ...)` became
  `from_dicts(cls, llm_args, vector_store_args, embeddings_args, ...)`
- `get_retriever_from_config(config, ...)` became
  `get_retriever_from_dicts(llm_args, vector_store_args, embeddings_args, ...)`

### 2c. Dead Code Removal

| File | Removed | Reason |
|------|---------|--------|
| `chains/declarative/runner.py` | `_is_likely_path()` static method, `import os` | Unused heuristic |
| `schemas/session.py` | `Conversation.to_dict()` | Exact duplicate of `to_list()` |

### 2d. DRY: Strategy Graph Compilation

Both `DeclarativeTeam` and `DeclarativeTeamRouter` had identical 35-line
if/elif/else blocks for mapping strategy types to graph builders, plus
duplicated `_STRATEGY_BUILDERS` dicts.

**Solution:** Extracted `compile_strategy_graph()` into `strategies.py`.

**`strategies.py`** (+70 lines):
- Added `STRATEGY_BUILDERS` dict (moved from team.py and router.py)
- Added `compile_strategy_graph()` — centralizes strategy-type to graph-builder
  dispatch, handles normalization (`"round-robin"` to `"round_robin"`, etc.)

**`team.py`** (-49 / +8): Replaced 35-line dispatch block + `_STRATEGY_BUILDERS`
dict with:
```python
self.compiled_graph = compile_strategy_graph(
    strategy_type=strategy_type,
    member_names=member_names,
    agents=self.agents,
    max_turns=max_turns,
    selector_spec=selector_spec,
    graph_spec=graph_spec,
)
```

**`router.py`** (-47 / +8): Same replacement.

---

## 3. Controller Removal — Delete ControllerClient, Config Class, Schema Cleanup

### Deleted: `controller_client.py` (-205 lines)

The entire `ControllerClient` class was removed. It was an HTTP client for the
genai-factory controller API with methods:

| Method | Purpose |
|--------|---------|
| `get_user()` | Look up user by username/email |
| `get_session()` | Load chat session |
| `update_session()` | Persist chat session |
| `get_project()` | Fetch project metadata |
| `create_workflow()` / `get_workflow()` / `update_workflow()` | Workflow CRUD |
| `get_data_source()` | Data source lookup |

All session functionality is now handled by `SessionStore` (artifact-backed).

### Rewritten: `config.py` (-51 / +20)

**Deleted** the `WorkflowServerConfig` class (Pydantic model, ~50 lines):
```python
# DELETED — this entire class
class WorkflowServerConfig(BaseModel):
    controller_url: str = "http://localhost:8001"
    controller_username: str = "guest"
    project_name: str = "default"
    verbose: bool = True
    log_level: str = "INFO"
    deployment_url: str = "http://localhost:8000"
    workflows_kwargs: dict[str, dict] = {}
    chunk_size: int = 1024
    chunk_overlap: int = 20
    embeddings: dict = {"class_name": "huggingface", ...}
    default_llm: dict = {"class_name": "langchain_openai.ChatOpenAI", ...}
    default_vector_store: dict = {"class_name": "milvus", ...}
```

**Kept** factory functions but changed their signatures to take plain dicts
instead of a config object:

| Function | Before | After |
|----------|--------|-------|
| `get_embedding_function` | `(config, embeddings_args=None)` | `(embeddings_args: dict)` |
| `get_llm` | `(config, llm_args=None)` | `(llm_args: dict)` |
| `get_vector_db` | `(config, collection_name, vector_store_args=None)` | `(vector_store_args: dict, embeddings_args: dict, collection_name=None)` |

### Updated: `__init__.py` (-2 / +1)

```python
# Before
from mlrun.agentic.config import WorkflowServerConfig, get_llm
from mlrun.agentic.controller_client import ControllerClient

# After
from mlrun.agentic.config import get_llm
```

### Updated: `chains/__init__.py` (+2 / -2)

Renamed export: `get_retriever_from_config` became `get_retriever_from_dicts`.

### Cleaned: `schemas/__init__.py` (-3 exports)

Removed: `APIDictResponse`, `APIResponse`, `OutputMode` (only used by the
controller).

### Cleaned: `schemas/base.py` (-29 / +1)

Removed three classes/enums that only existed for the controller API:

```python
# DELETED
class APIResponse(BaseModel):        # success/data/error pattern
class APIDictResponse(APIResponse):  # dict variant
class OutputMode(str, Enum):         # NAMES/SHORT/DICT/DETAILS
```

Removed unused imports: `Enum`, `HTTPException`, `Any`.

### Fixed: `schemas/workflow.py` (+2 / -2)

1. `project_id: str` became `project_id: Optional[str] = None` — workflows can
   exist without a controller-assigned project_id.
2. Fixed bug in `WorkflowEvent.to_dict()`: `"session_id": self.session_id`
   became `"session_name": self.session_name` (the attribute was `session_name`,
   not `session_id`).

---

## Architectural Principles

1. **No controller dependency** — `controller_client.py` is deleted. All
   functionality is self-contained within MLRun.

2. **No config class** — `WorkflowServerConfig` is deleted. `config.py`
   contains only stateless factory functions that take plain dicts.

3. **No `context._config`** — nothing reads it anymore. `graph_initializer`
   only sets `context.session_store`.

4. **Self-sufficient chains** — every chain creates its own LLM via a lazy
   `@property`. No shared config injection needed.

5. **Pluggable SessionStore** — the interface (`read_state` + `save`) is
   backend-agnostic. Current: MLRun artifacts. Future: Memory Gateway.

6. **DRY strategy compilation** — `compile_strategy_graph()` in `strategies.py`
   is shared by both `DeclarativeTeam` and `DeclarativeTeamRouter`.
