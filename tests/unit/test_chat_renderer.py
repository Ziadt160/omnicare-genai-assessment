"""The chat renderer.

The frontend is static files with no JS toolchain, and adding one for a
hundred lines of markup would cost more than it is worth - so the real
`app.js` is loaded under a minimal DOM stub in node and its exported
`renderText` is called directly. See `js/render_harness.mjs`.

Skipped rather than failed when node is absent: a Python developer without it
should still get a green suite, and CI has it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
HARNESS = Path(__file__).parent / "js" / "render_harness.mjs"

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node is not installed"
)


def test_the_renderer_turns_markdown_into_markup() -> None:
    """Reported with a screenshot: an answer arrived with "### Section 1" and
    "- **Coverage**:" shown literally, hashes and hyphens and all.

    The harness names every failed check, so a regression here says which
    property broke rather than only that something did.
    """
    result = subprocess.run(
        ["node", str(HARNESS)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr or result.stdout
