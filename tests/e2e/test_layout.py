"""Layout invariants that only break once there is real content on the page.

Both of these were found by using the app rather than by reading it, and
neither is visible on a fresh load - which is exactly why they need a test. A
screenshot of an empty conversation looks correct in both cases.

    docker compose up -d
    pytest tests/e2e/test_layout.py -m e2e
"""

from __future__ import annotations

import os

import httpx
import pytest

FRONTEND = os.environ.get("FRONTEND_URL", "http://127.0.0.1:3000")


def _reachable() -> bool:
    try:
        return httpx.get(FRONTEND, timeout=3).status_code == 200
    except Exception:
        return False


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not _reachable(), reason=f"no frontend at {FRONTEND}"),
]


@pytest.fixture(scope="module")
def browser():
    """One browser for the module - see the note in test_voice_ui.py."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            b = pw.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:
            pytest.skip(f"no system Chrome to drive: {exc}")
        yield b
        b.close()


@pytest.fixture()
def page(browser):
    p = browser.new_page(viewport={"width": 1000, "height": 700})
    p.goto(FRONTEND, wait_until="networkidle")
    p.wait_for_function("() => window.OmniCare !== undefined", timeout=10_000)
    yield p
    p.close()


def fill_thread(page, count: int = 40) -> None:
    page.evaluate(
        """n => {
            for (let i = 0; i < n; i++) {
                window.OmniCare.addMessage('user', 'Question number ' + i + '?');
                window.OmniCare.addMessage(
                    'assistant',
                    'A reasonably long answer number ' + i + ', of the length a ' +
                    'coverage explanation actually runs to, so the thread grows ' +
                    'the way it does in use rather than in a contrived way.'
                );
            }
        }""",
        count,
    )
    page.wait_for_timeout(250)


# ------------------------------------------------------- the composer stays put

@pytest.mark.parametrize(
    ("width", "height"),
    [(1000, 700), (390, 780), (1440, 900)],   # desktop, phone, large desktop
)
def test_the_composer_stays_on_screen_however_long_the_thread_gets(
    page, width: int, height: int
) -> None:
    """The bug: `min-height: 100vh` on a grid whose rows are `auto 1fr auto auto`.

    `min-height` lets the body grow past the viewport, and a grid item's default
    `min-height: auto` stops the thread row from shrinking below its content -
    so instead of the thread scrolling, the whole page grew and pushed the input
    box off the bottom. Invisible until a conversation is long enough, which is
    to say invisible in every screenshot.
    """
    page.set_viewport_size({"width": width, "height": height})
    fill_thread(page)

    box = page.evaluate(
        """() => {
            const r = document.getElementById('composer').getBoundingClientRect();
            return {top: r.top, bottom: r.bottom, height: r.height};
        }"""
    )
    assert box["height"] > 0, "the composer collapsed"
    assert box["bottom"] <= height + 1, (
        f"the input box is {box['bottom'] - height:.0f}px below the fold "
        f"at {width}x{height}"
    )


def test_the_page_itself_does_not_scroll(page) -> None:
    """The thread scrolls; the page does not. Otherwise the composer and the
    masthead slide away with the content."""
    fill_thread(page)
    overflow = page.evaluate(
        "() => document.documentElement.scrollHeight - window.innerHeight"
    )
    assert overflow <= 1, f"the document scrolls by {overflow}px"


def test_the_thread_is_what_scrolls(page) -> None:
    fill_thread(page)
    scrollable = page.evaluate(
        """() => {
            const t = document.getElementById('thread');
            return t.scrollHeight > t.clientHeight + 1;
        }"""
    )
    assert scrollable, "the thread should be the scrolling element"


def test_the_newest_message_is_the_one_you_can_see(page) -> None:
    """A conversation that does not follow itself down is unusable: the answer
    arrives off-screen and looks like nothing happened."""
    fill_thread(page)
    page.evaluate("() => window.OmniCare.addMessage('assistant', 'THE LATEST ANSWER')")
    page.wait_for_timeout(200)
    visible = page.evaluate(
        """() => {
            const t = document.getElementById('thread');
            const last = t.lastElementChild.getBoundingClientRect();
            const box = t.getBoundingClientRect();
            return last.bottom <= box.bottom + 2 && last.top >= box.top - 2;
        }"""
    )
    assert visible, "the newest message is not in view"
