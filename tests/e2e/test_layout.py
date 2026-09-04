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


# ------------------------------------------------------------------ contrast

CONTRAST_PAIRS = [
    ("the conversation", ".bubble", "color", "backgroundColor"),
    ("your own messages", ".msg--user .bubble", "color", "backgroundColor"),
    ("the send button", "#send", "color", "backgroundColor"),
    ("the citation file name", ".cite__file", "color", "backgroundColor"),
]


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_text_is_readable_in_both_themes(browser, scheme: str) -> None:
    """White text was hard-coded onto the accent and the stop colour.

    That was already failing in dark mode before the palette changed - the old
    dark accent was a light amber, so white on it measured about 1.9:1 - and it
    is the kind of thing that is invisible to whoever built it, because they
    are looking at the light theme.
    """
    page = browser.new_page(color_scheme=scheme,
                            viewport={"width": 1000, "height": 700})
    try:
        page.goto(FRONTEND, wait_until="networkidle")
        page.evaluate(
            """() => {
                window.OmniCare.addMessage('user', 'Is flood damage covered?');
                window.OmniCare.addMessage('assistant', 'Covered up to $25,000.',
                  {sources: ['sample_policy.md § Section 1: Home Water Damage Coverage']});
            }"""
        )
        page.wait_for_timeout(200)

        results = page.evaluate(
            r"""pairs => {
                const lum = (c) => {
                    const [r, g, b] = c.match(/\d+/g).slice(0, 3).map(Number);
                    const f = (v) => {
                        v /= 255;
                        return v <= 0.04045 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
                    };
                    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b);
                };
                // Walk up for the nearest painted ancestor: a transparent
                // background is the parent's, not black.
                const bgOf = (el) => {
                    for (let n = el; n; n = n.parentElement) {
                        const c = getComputedStyle(n).backgroundColor;
                        if (c && !c.startsWith('rgba(0, 0, 0, 0)')) return c;
                    }
                    return 'rgb(255,255,255)';
                };
                return pairs.map(([label, sel]) => {
                    const el = document.querySelector(sel);
                    if (!el) return [label, null];
                    const a = lum(getComputedStyle(el).color), b = lum(bgOf(el));
                    const [hi, lo] = a > b ? [a, b] : [b, a];
                    return [label, (hi + 0.05) / (lo + 0.05)];
                });
            }""",
            CONTRAST_PAIRS,
        )
        poor = [(label, round(r, 2)) for label, r in results if r is not None and r < 4.5]
        assert not poor, f"unreadable in {scheme} mode: {poor}"
    finally:
        page.close()


# ------------------------------------------------- the citation stays in view

def stream_answer(page, paragraphs: int = 4) -> None:
    """Drive the real gateway event sequence for one streamed answer."""
    page.evaluate(
        """n => {
            const H = window.OmniCare.handleEvent;
            H({type: 'started', payload: {}});
            H({type: 'tool_end', payload: {name: 'search_policy_documents', status: 'ok'}});
            for (let i = 0; i < n; i++) {
                H({type: 'token', payload: {text:
                    'Sudden pipe bursts are covered up to $25,000 with a $500 ' +
                    'deductible, while gradual leaks and flood damage are ' +
                    'strictly excluded. Paragraph ' + i + '.' + String.fromCharCode(10, 10)}});
            }
            H({type: 'sources', payload: {sources: [
                'sample_policy.md § Section 1: Home Water Damage Coverage',
                'sample_policy.md § Section 2: Personal Property Protection']}});
            H({type: 'done', payload: {}});
        }""",
        paragraphs,
    )
    page.wait_for_timeout(300)


def test_the_citation_is_on_screen_after_a_streamed_answer(page) -> None:
    """The bug behind "the citation source isn't showing".

    `scrollTop` was set when a message was added and on every token, but the
    citations arrive *after* the last token - and `sources` appended them
    without scrolling. So the block existed, was `display: flex`, was fully
    opaque, and sat below the fold: present in the DOM and invisible on screen,
    which is the worst way for it to fail. The longer the answer, the further
    down it went.
    """
    fill_thread(page, 6)
    stream_answer(page)

    visible = page.evaluate(
        """() => {
            const s = [...document.querySelectorAll('.msg--assistant .sources')].pop();
            if (!s) return null;
            const r = s.getBoundingClientRect();
            const t = document.getElementById('thread').getBoundingClientRect();
            return {below: Math.round(r.bottom - t.bottom), h: Math.round(r.height)};
        }"""
    )
    assert visible is not None, "no citations were rendered at all"
    assert visible["below"] <= 2, (
        f"the citation block is {visible['below']}px below the visible area"
    )


def test_the_tool_chip_is_on_screen_too(page) -> None:
    """Same mechanism: `tool_end` appended without scrolling."""
    fill_thread(page, 6)
    stream_answer(page)

    below = page.evaluate(
        """() => {
            const c = [...document.querySelectorAll('.msg--assistant .tools')].pop();
            const t = document.getElementById('thread').getBoundingClientRect();
            return Math.round(c.getBoundingClientRect().bottom - t.bottom);
        }"""
    )
    assert below <= 2, f"the tool chip is {below}px below the visible area"


def test_reading_back_through_history_is_not_interrupted(page) -> None:
    """The fix must not become a different annoyance: someone scrolled up to
    re-read an earlier answer should not be yanked to the bottom when the next
    citation arrives."""
    fill_thread(page, 20)
    page.evaluate("() => { document.getElementById('thread').scrollTop = 0; }")
    page.wait_for_timeout(100)
    stream_answer(page)

    top = page.evaluate("() => document.getElementById('thread').scrollTop")
    assert top < 200, f"the reader was dragged down to {top}px"
