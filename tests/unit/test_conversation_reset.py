""""New conversation" has to actually start one.

Reported by a user and reproduced against the running stack: the button
emptied the screen and nothing else. The next message came back answered with
context the policyholder could no longer see - "What did I just tell you
about?" described the burst pipe from the conversation they had just cleared.

`resetConversation` set the id to null, and a request with no conversation id
does not start a new conversation: `ConversationStore.ensure` resolves it to
the user's *most recent* one, deliberately, so the graded request schema (which
has no conversation_id field) can still hold a multi-turn thread. Dropping the
id resolved straight back to the conversation being abandoned.

Driven through node against the real `app.js` - see `js/reset_harness.mjs`.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
HARNESS = Path(__file__).parent / "js" / "reset_harness.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def test_reset_starts_a_conversation_the_server_has_not_seen() -> None:
    """The harness names every failed check, so a regression says which
    property broke rather than only that something did."""
    result = subprocess.run(
        ["node", str(HARNESS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr or result.stdout
