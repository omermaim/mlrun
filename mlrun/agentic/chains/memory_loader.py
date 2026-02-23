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

"""Chain step that loads long-term memory facts into the workflow event."""

from mlrun.agentic.chains.base import ChainRunner
from mlrun.agentic.schemas import WorkflowEvent
from mlrun.agentic.utils import logger


class MemoryLoader(ChainRunner):
    """Loads relevant facts from the memory store into event.state.

    Placed after ``SessionLoader``, before chains that need personalization.
    Gracefully no-ops if ``context.memory_store`` is not configured.

    :param entity_key_template: Template for entity key (supports ``{username}``).
    :param limit: Max facts to retrieve per query.
    """

    def __init__(self, entity_key_template="user:{username}", limit=10, **kwargs):
        super().__init__(**kwargs)
        self.entity_key_template = entity_key_template
        self.limit = limit

    def _run(self, event: WorkflowEvent):
        memory_store = getattr(self.context, "memory_store", None)
        if not memory_store:
            return {}

        entity_key = self.entity_key_template.format(
            username=event.username or "guest",
        )
        hits = memory_store.retrieve(
            entity_key=entity_key,
            query=event.query,
            limit=self.limit,
        )

        facts = []
        for hit in hits:
            for fact in hit.record.facts:
                facts.append(
                    {
                        "key": fact.key,
                        "value": fact.value,
                        "confidence": fact.confidence,
                    }
                )

        event.state["memory_context"] = facts
        logger.debug(
            "Loaded memory context",
            entity_key=entity_key,
            num_facts=len(facts),
        )
        return {}
