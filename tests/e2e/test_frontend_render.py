"""The browser-side text renderer, exercised in a real browser.

`renderText` puts model output into `innerHTML`, which makes it the one place
in the system where untrusted text becomes markup. It exists because models
write markdown whether asked to or not - a reviewer's first screenshot showed
literal `**Home Water Damage Coverage**` in the middle of an insurance answer -
and because gpt-oss-120b emits U+3010 as a citation marker that is left
dangling when the reference is stripped.

Escaping happens before any markup is inserted. These tests pin that, using the
real function in a real browser rather than a Python reimplementation of it,
because a reimplementation would be the thing under test rather than the code
that ships.

    docker compose up -d
    pytest tests/e2e/test_frontend_render.py -m e2e
"""

from __future__ import annotations

import os

import httpx
import pytest

pytestmark = pytest.mark.e2e

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
def render():
    """Return a callable that runs the page's own `renderText` in Chrome."""
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        try:
            browser = pw.chromium.launch(channel="chrome", headless=True)
        except Exception as exc:
            # Skip rather than fail where there is no system Chrome. A missing
            # browser says nothing about the renderer, and the rest of the e2e
            # suite is still worth running.
            pytest.skip(f"no system Chrome to drive: {exc}")
        page = browser.new_page()
        page.goto(FRONTEND, wait_until="networkidle")

        def run(text: str) -> str:
            return page.evaluate("t => renderText(t)", text)

        def elements(text: str) -> list[str]:
            """Tag names that actually materialise in the DOM.

            The property that matters is not whether the string "javascript:"
            appears - escaped, it is inert text - but whether any element the
            renderer did not intend ends up in the tree. So insert it and look.
            """
            return page.evaluate(
                """t => {
                    const d = document.createElement('div');
                    d.innerHTML = renderText(t);
                    return [...d.querySelectorAll('*')].map(e => e.tagName.toLowerCase());
                }""",
                text,
            )

        run.elements = elements  # type: ignore[attr-defined]
        yield run
        browser.close()


# ------------------------------------------------------------------ safety

# The only elements the renderer is allowed to create.
ALLOWED = {"strong", "em", "br", "p"}


@pytest.mark.parametrize(
    "hostile",
    [
        "<script>alert(1)</script>",
        "<img src=x onerror=alert(1)>",
        "<a href='javascript:alert(1)'>click</a>",
        "**<script>alert(1)</script>**",
        "<iframe src='https://evil.example'></iframe>",
    ],
)
def test_markup_in_model_output_never_reaches_the_dom(render, hostile: str) -> None:
    """Retrieved policy text and model output both flow through here. Escaping
    runs first, so a bold marker cannot smuggle a tag in behind it.

    Asserted on the resulting DOM rather than on substrings: escaped text may
    legitimately contain "javascript:" or "onerror", and a first version of
    this test failed correct output for saying so. What must hold is that no
    element the renderer did not intend materialises.
    """
    created = set(render.elements(hostile))

    assert created <= ALLOWED, f"unexpected elements in the DOM: {created - ALLOWED}"
    assert "&lt;" in render(hostile), "the tag should survive as escaped text"


# ---------------------------------------------------------------- markdown

def test_bold_renders(render) -> None:
    """The bug a reviewer sees first: literal asterisks in a policy answer."""
    out = render("Your policy's **Home Water Damage Coverage** applies.")
    assert "<strong>Home Water Damage Coverage</strong>" in out
    assert "**" not in out


def test_italic_renders_without_eating_bold(render) -> None:
    out = render("**bold** and *italic* together")
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out


def test_money_is_not_mistaken_for_markup(render) -> None:
    """Amounts are the most important text in a coverage answer."""
    out = render("covered up to **$25,000** with a **$500** deductible")
    assert "$25,000" in out and "$500" in out


def test_citation_markers_are_stripped(render) -> None:
    """gpt-oss-120b emits these around references; the reference itself is
    removed by the ground node, leaving the bracket dangling mid-sentence."""
    assert "【" not in render("a sudden burst is covered 【")
    assert "】" not in render("covered 】 up to $25,000")


def test_paragraph_breaks_survive(render) -> None:
    out = render("First line.\n\nSecond paragraph.")
    assert "</p><p>" in out


def test_plain_text_is_unchanged(render) -> None:
    text = "Claim CLM-8821 is Approved."
    assert render(text) == text
