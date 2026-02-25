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

from enum import Enum
from typing import Any, Optional

from mlrun.agentic.schemas.base import BaseWithVerMetadata
from mlrun.agentic.schemas.session import Conversation


class WorkflowType(str, Enum):
    INGESTION = "ingestion"
    APPLICATION = "application"
    DATA_PROCESSING = "data-processing"
    TRAINING = "training"
    EVALUATION = "evaluation"


class Workflow(BaseWithVerMetadata):
    _top_level_fields = ["workflow_type"]

    workflow_type: WorkflowType
    project_id: Optional[str] = None
    deployment: Optional[str] = None
    workflow_function: Optional[str] = None
    configuration: Optional[dict] = None
    graph: Optional[dict] = None


class WorkflowEvent:
    """A workflow event that flows through the agentic chain graph."""

    def __init__(
        self,
        query: Optional[str] = None,
        username: Optional[str] = None,
        session_name: Optional[str] = None,
        **kwargs,
    ):
        self.username: Optional[str] = username
        self.session_name: Optional[str] = session_name
        self.original_query: Optional[str] = query
        self.query: Optional[str] = query
        self.kwargs: dict = kwargs

        self.session: Any = None
        self.results: dict = {}
        self.state: dict = {}
        self.memory_context: list[dict] = []
        self.conversation: Conversation = Conversation()

    def to_dict(self) -> dict:
        session_data = None
        if self.session:
            if hasattr(self.session, "to_dict"):
                session_data = self.session.to_dict()
            else:
                # agentstores Session handle — serialize as session_id only
                session_data = {"session_id": getattr(self.session, "session_id", None)}
        return {
            "username": self.username,
            "session_name": self.session_name,
            "query": self.query,
            "kwargs": self.kwargs,
            "results": self.results,
            "state": self.state,
            "memory_context": self.memory_context,
            "conversation": self.conversation.to_list(),
            "session": session_data,
        }

    def __getitem__(self, item):
        return getattr(self, item)

    def __str__(self) -> str:
        return self.results.get("answer", "")

    def __repr__(self) -> str:
        return (
            f"WorkflowEvent(query={self.query!r}, username={self.username!r}, "
            f"session_name={self.session_name!r}, "
            f"results_keys={list(self.results.keys())})"
        )
