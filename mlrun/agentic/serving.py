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

"""Default graph_initializer that wires session and memory stores via env vars."""

import os
import mlrun.utils
from agentstores import Client
from mlrun.agentic.sessions import SessionStore
logger = mlrun.utils.logger


def graph_initializer(server):
    """Set up session and memory stores on the serving context.

    Reads configuration from environment variables:

    - ``AGENTIC_DB_PATH``      – local SQLite path (default ``./agentstores.db``).
    - ``AGENTIC_GATEWAY_URL``  – remote gateway URL (takes precedence over db_path).
    - ``AGENTIC_ENABLE_FTS``   – enable full-text search (``1`` to enable).

    Idempotent: skips if ``context.session_store`` is already set.
    """
    context = server.context
    if getattr(context, "session_store", None) is not None:
        return

    gateway_url = os.environ.get("AGENTIC_GATEWAY_URL")
    db_path = os.environ.get("AGENTIC_DB_PATH", "./agentstores.db")
    enable_fts = os.environ.get("AGENTIC_ENABLE_FTS", "0") == "1"

    if gateway_url:
        client = Client(base_url=gateway_url)
    else:
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        client = Client(db_path=db_path, enable_fts=enable_fts)

    as_sessions = client.session_stores.create("sessions", if_exists="return")
    context.session_store = SessionStore(session_store=as_sessions)

    context.memory_store = client.mem_stores.create("memory", if_exists="return")

    logger.info(
        "Agentic stores initialized",
        gateway=gateway_url or f"local ({db_path})",
    )
