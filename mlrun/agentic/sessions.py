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

"""Session persistence via agentstores."""

from agentstores.client.session import SessionEvent
from agentstores.client.session import SessionStore as ASSessionStore

import mlrun.errors
from mlrun.agentic.schemas.session import Conversation, Message
from mlrun.agentic.schemas.workflow import WorkflowEvent
from mlrun.agentic.utils import logger

# Role mapping: mlrun ChatRole <-> agentstores EventRole
_ROLE_TO_EVENT = {
    "Human": "user",
    "User": "user",
    "AI": "assistant",
    "Agent": "assistant",
    "System": "system",
}
_EVENT_TO_ROLE = {
    "user": "Human",
    "assistant": "AI",
    "system": "System",
    "tool": "System",
    "other": "System",
}


class SessionStore:
    """Manages chat session persistence via agentstores.

    The public interface is ``read_state(event)`` and ``save(event)``.
    ``SessionLoader`` and ``HistorySaver`` rely only on these two methods.

    :param session_store: An agentstores ``SessionStore`` instance.
    :param default_username: Fallback username when none is on the event.
    """

    def __init__(self, session_store: ASSessionStore, default_username="guest"):
        self._session_store = session_store
        self.default_username = default_username

    def read_state(self, event: WorkflowEvent):
        """Load session state into the workflow event.

        :param event: WorkflowEvent to populate with session data.
        """
        event.username = event.username or self.default_username

        if not event.session and event.session_name:
            try:
                session = self._session_store.get(event.session_name)
                events = session.read_events(kinds=["message"], order="asc")
                event.conversation = self._events_to_conversation(events)
            except Exception as exc:
                logger.debug(
                    "Session not found, creating new",
                    session_name=event.session_name,
                    error=mlrun.errors.err_to_str(exc),
                )
                session = self._session_store.create(
                    session_id=event.session_name,
                    user_id=event.username,
                )
                event.conversation = Conversation()

            event.session = session  # Store handle for save()

    def save(self, event: WorkflowEvent):
        """Persist session state.

        :param event: WorkflowEvent containing session data to save.
        """
        if event.session_name and event.session:
            new_messages = event.conversation.messages[event.conversation.saved_index :]
            if new_messages:
                as_events = self._messages_to_events(new_messages)
                event.session.update(append_events=as_events, commit=True)
                event.conversation.saved_index = len(event.conversation.messages)

    @staticmethod
    def _messages_to_events(messages):
        """Convert mlrun Message objects to agentstores SessionEvent objects."""
        events = []
        for msg in messages:
            role = _ROLE_TO_EVENT.get(msg.role.value, "other")
            extra = {}
            if msg.sources:
                extra["sources"] = msg.sources
            events.append(
                SessionEvent(
                    kind="message",
                    role=role,
                    content=msg.content,
                    extra_data=extra,
                    human_feedback=msg.human_feedback,
                )
            )
        return events

    @staticmethod
    def _events_to_conversation(events):
        """Convert agentstores SessionEvent objects to a mlrun Conversation."""
        messages = []
        for ev in events:
            role = _EVENT_TO_ROLE.get(ev.role, "Human")
            sources = ev.extra_data.get("sources") if ev.extra_data else None
            messages.append(
                Message(
                    role=role,
                    content=ev.content,
                    sources=sources,
                    human_feedback=ev.human_feedback,
                )
            )
        conv = Conversation(messages=messages)
        conv.saved_index = len(messages)
        return conv
