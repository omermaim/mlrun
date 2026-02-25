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

"""Chain step that extracts and saves facts to long-term memory."""

import json

from agentstores.client.longterm import Fact, Provenance, SourceRef
from langchain_openai import ChatOpenAI

import mlrun.errors
from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.schemas import WorkflowEvent
from mlrun.agentic.utils import logger


class MemorySaver(ChainRunner):
    """Saves facts to long-term memory store.

    Two modes:

    - ``extract=False`` (default): reads facts from ``event.state["new_facts"]``.
      Only saves if an upstream step explicitly provided facts.
    - ``extract=True``: uses LLM to auto-extract facts from the Q&A exchange.

    Gracefully no-ops if ``context.memory_store`` is not configured.

    :param extract: Whether to use LLM extraction (False = explicit facts only).
    :param model_name: LLM model name used for extraction when ``extract=True``.
    """

    def __init__(self, extract=False, model_name="gpt-4", **kwargs):
        super().__init__(**kwargs)
        self.extract = extract
        self._model_name = model_name
        self._llm = None

    def post_init(
        self,
        mode="sync",
        context=None,
        namespace=None,
        creation_strategy=None,
        **kwargs,
    ):
        if self.extract and not self._llm:
            self._llm = ChatOpenAI(model=self._model_name, temperature=0)

    def _run(self, event: WorkflowEvent):
        memory_store = getattr(self.context, "memory_store", None)
        if not memory_store:
            return {}

        entity_key = f"user:{event.username or 'guest'}"

        if self.extract:
            facts = self._extract_facts_with_llm(event)
        else:
            facts = event.state.get("new_facts", [])

        if facts:
            logger.debug(
                "Saving facts to LTM",
                entity_key=entity_key,
                num_facts=len(facts),
            )
            memory_store.write(
                key=entity_key,
                content=event.results.get("answer", ""),
                facts=facts,
                provenance=Provenance(
                    sources=[
                        SourceRef(
                            kind="session",
                            ref_id=event.session_name or "",
                        )
                    ],
                    extractor="memory_saver_v1",
                ),
                commit=True,
            )
        return {}

    def _extract_facts_with_llm(self, event):
        """Extract user facts from the Q&A exchange using an LLM.

        :param event: WorkflowEvent containing the query and answer.
        :return: List of ``Fact`` objects, or empty list on failure.
        """
        prompt = (
            "Extract factual information about the user from this exchange.\n"
            "Return ONLY a JSON array of facts. Each fact: kind, key, value, "
            "confidence (0-1).\n\n"
            f"User said: {event.original_query}\n"
            f"Assistant replied: {event.results.get('answer', '')}\n\n"
            "If no facts can be extracted, return []."
        )
        response = self._llm.invoke(prompt)
        try:
            raw = json.loads(response.content)
            return [Fact(**f) for f in raw]
        except Exception as exc:
            logger.warning(
                "Failed to parse LLM fact extraction response",
                error=mlrun.errors.err_to_str(exc),
            )
            return []
