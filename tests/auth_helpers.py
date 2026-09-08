"""Persistent cookie fixtures for service tests; production has no test auth bypass."""

import hashlib
import time

from medsegagent.auth import SESSION_COOKIE
from medsegagent.web import create_app

ALICE = {"Cookie": f"{SESSION_COOKIE}=" + "a" * 43}
BOB = {"Cookie": f"{SESSION_COOKIE}=" + "b" * 43}


def create_test_app(root, public_url="http://localhost"):
    app = create_app(root, public_url)
    db = app.state.service.db
    with db:
        for principal, value in (("alice", "a" * 43), ("bob", "b" * 43)):
            db.execute(
                "INSERT OR IGNORE INTO auth_principals VALUES(?,?,?,?)",
                (principal, "guest", principal, time.time()),
            )
            db.execute(
                "INSERT OR REPLACE INTO auth_sessions VALUES(?,?,?)",
                (hashlib.sha256(value.encode()).hexdigest(), principal, time.time() + 3600),
            )
    return app
