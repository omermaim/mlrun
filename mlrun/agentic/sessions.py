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

import json

import mlrun.agentic.schemas as schemas
from mlrun.agentic.utils import logger


class SessionStore:
    """Manages chat session persistence via MLRun artifacts.

    The public interface is ``read_state(event)`` and ``save(event)``.
    ``SessionLoader`` and ``HistorySaver`` rely only on these two methods.

    :param project: MLRun project for artifact-backed storage.
    :param default_username: Fallback username when none is on the event.
    """

    def __init__(self, project, default_username="guest"):
        self._project = project
        self.default_username = default_username

    @staticmethod
    def _artifact_key(session_name: str) -> str:
        """Generate artifact key for a session."""
        return f"session-{session_name}"

    def read_state(self, event: schemas.WorkflowEvent):
        """Load session state into the workflow event.

        :param event: WorkflowEvent to populate with session data.
        """
        event.username = event.username or self.default_username

        if not event.session and event.session_name:
            artifact_key = self._artifact_key(event.session_name)
            try:
                artifact = self._project.get_artifact(artifact_key)
                body = artifact.get_body()
                session_data = (
                    json.loads(body) if isinstance(body, (str, bytes)) else body
                )
                event.session = schemas.ChatSession(**session_data)
                event.conversation = event.session.to_conversation()
            except Exception:
                logger.debug(
                    "Session %s not found, starting fresh",
                    event.session_name,
                )
                event.session = schemas.ChatSession(
                    name=event.session_name,
                    owner_id=event.username or self.default_username,
                    workflow_id=event.workflow_id or "",
                )
                event.conversation = schemas.Conversation()

    def save(self, event: schemas.WorkflowEvent):
        """Persist session state.

        :param event: WorkflowEvent containing session data to save.
        """
        if event.session_name:
            session_data = {
                "name": event.session_name,
                "owner_id": event.username or self.default_username,
                "history": event.conversation.to_list(),
                "workflow_id": event.workflow_id or "",
            }
            artifact_key = self._artifact_key(event.session_name)
            self._project.log_artifact(
                artifact_key,
                body=json.dumps(session_data).encode("utf-8"),
                labels={
                    "kind": "chat-session",
                    "session_owner": event.username or self.default_username,
                },
            )
