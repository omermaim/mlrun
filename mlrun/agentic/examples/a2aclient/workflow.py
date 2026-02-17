# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A2A Client Workflow — programmatic chain graph using mlrun.agentic.

This workflow demonstrates how to build a conversational DAG with intent-based
routing using mlrun.agentic chains and MLRun's serving graph — no external
controller or workflow_server needed.

Architecture::

    SessionLoader → RefineQuery → IntentClassifier → IntentChoice
                                                         ├→ A2AClient (meeting)
                                                         └→ Communicator (casual)
                                                              ↓
    LanguageGuardrail → HallucinationGuardrail → HistorySaver → respond()

Usage — local mock::

    python -c "
    from workflow import build_serving_function, create_mock, infer
    build_serving_function()
    create_mock()
    print(infer('Summarize this meeting transcript: ...'))
    "

Usage — deploy to cluster::

    from workflow import build_serving_function, deploy
    fn = build_serving_function()
    deploy()
"""

import os
from pathlib import Path

from dotenv import load_dotenv

import mlrun
import mlrun.serving as mlrun_serving
from mlrun.agentic.chains.a2a_client import A2AClient
from mlrun.agentic.chains.base import HistorySaver, SessionLoader
from mlrun.agentic.chains.communicator import Communicator
from mlrun.agentic.chains.hallucination_guardrail import HallucinationGuardrail
from mlrun.agentic.chains.intent_choice import IntentChoice
from mlrun.agentic.chains.intent_classifier import IntentClassifier
from mlrun.agentic.chains.language_guardrail import LanguageGuardrail
from mlrun.agentic.chains.refine import (
    CONVERSATION_CONTEXT_REFINER_PROMPT,
    RefineQuery,
)
from mlrun.agentic.sessions import SessionStore

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_current_dir = Path(__file__).parent.resolve()
load_dotenv(dotenv_path=_current_dir / ".env", override=True)

PROJECT_NAME = os.getenv("MLRUN_PROJECT_NAME", "a2aclient-demo")

# ---------------------------------------------------------------------------
# Module-level state (populated by build / create_mock / deploy)
# ---------------------------------------------------------------------------
_serving_function = None
_mock_server = None


def graph_initializer(server: mlrun_serving.GraphServer):
    """Called once when the serving graph starts up.

    Sets the session store (artifact-backed) on the server context
    so that ``SessionLoader`` and ``HistorySaver`` can read/write sessions.
    """
    context = server.context
    if getattr(context, "session_store", None) is None:
        project = mlrun.get_or_create_project(PROJECT_NAME)
        context.session_store = SessionStore(project=project)


def build_serving_function(
    name="a2a-workflow",
    image=None,
    requirements=None,
):
    """Build an MLRun serving function with the A2A workflow graph.

    :param name:         Serving function name.
    :param image:        Docker image (default mlrun/mlrun).
    :param requirements: Extra pip requirements.
    :return: The MLRun serving function object.
    """
    global _serving_function

    project = mlrun.get_or_create_project(PROJECT_NAME)

    default_reqs = [
        "a2a-sdk==0.3.0",
        "httpx>=0.24,<1.0",
        "langchain",
        "langchain-openai",
        "openai",
        "python-dotenv",
    ]

    _serving_function = project.set_function(
        name=name,
        func=str(_current_dir / "workflow.py"),
        kind="serving",
        image=image or "mlrun/mlrun",
        requirements=requirements or default_reqs,
    )
    _serving_function.spec.graph_initializer = "graph_initializer"

    # -- Build the DAG --
    root = _serving_function.set_topology("flow", engine="async")

    # Chain instances
    session_loader = SessionLoader(name="session-loader")
    refine_query = RefineQuery(
        name="refine-query",
        prompt_template=CONVERSATION_CONTEXT_REFINER_PROMPT,
    )
    intent_classifier = IntentClassifier(name="intent-classifier")
    intent_choice = IntentChoice(name="intent-choice")
    a2a_client = A2AClient(
        name="Atomic Agent(a2a)",
        base_url=os.getenv("A2A_BASE_URL", "http://localhost:10000"),
    )
    communicator = Communicator(name="communicator")
    language_guardrail = LanguageGuardrail(name="language-guardrail")
    hallucination_guardrail = HallucinationGuardrail(name="hallucination-guardrail")
    history_saver = HistorySaver(name="history-saver")

    # First part: session → refine → classify → choice
    classify_task = (
        root.to(session_loader).to(refine_query).to(intent_classifier)
    )
    choice_task = classify_task.to(intent_choice)

    # Choice branches
    choice_task.to(a2a_client)
    choice_task.to(communicator)

    # Merge branches back, then guardrails → save → respond
    language_guardrail_task = root.add_step(
        language_guardrail, after=["Atomic Agent(a2a)", "communicator"]
    )
    language_guardrail_task.to(hallucination_guardrail).to(history_saver).respond()

    return _serving_function


def create_mock():
    """Create a local mock server for testing (no cluster needed).

    :return: Mock server instance.
    """
    global _mock_server
    if _serving_function is None:
        raise RuntimeError("Call build_serving_function() first")
    _mock_server = _serving_function.to_mock_server()
    return _mock_server


def deploy(**kwargs):
    """Deploy the serving function to the cluster.

    :return: Deployment URL.
    """
    if _serving_function is None:
        raise RuntimeError("Call build_serving_function() first")
    return _serving_function.deploy(**kwargs)


def infer(query: str, username=None, session_name=None):
    """Run a query through the workflow.

    :param query:        User query string.
    :param username:     Optional username for session tracking.
    :param session_name: Optional session name for conversation continuity.
    :return: Answer string.
    """
    event = {"query": query}
    if username:
        event["username"] = username
    if session_name:
        event["session_name"] = session_name

    if _mock_server:
        resp = _mock_server.test("", body=event)
    elif _serving_function:
        resp = _serving_function.invoke("", body=event)
    else:
        raise RuntimeError("Call create_mock() or deploy() first")

    if hasattr(resp, "results"):
        return resp.results.get("answer", str(resp))
    return resp.get("answer", str(resp)) if isinstance(resp, dict) else str(resp)
