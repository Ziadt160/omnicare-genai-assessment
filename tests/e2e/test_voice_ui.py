"""The voice surface, exercised in a real browser.

Two things are pinned here.

**The transcript.** A call used to leave no readable record of what the
assistant said: the caller's words were shown, the tool chips were shown and
the citations were shown, but the answer itself existed only as audio. Hang up
and there was nothing to re-read. These tests feed the real data-channel
messages to the real handler and assert on the resulting DOM.

**The orb.** It is driven by an `AnalyserNode` on the actual audio rather than
by a timer, which is the point: a decorative animation looks identical whether
or not media is flowing, and media failing silently - room connected, no sound -
is exactly how WebRTC through Docker goes wrong.

Driving a genuine call would need a microphone, an SFU and a model; what is
worth testing here is the rendering contract, so the messages are injected.

    docker compose up -d
    pytest tests/e2e/test_voice_ui.py -m e2e
"""

from __future__ import annotations

import os

import httpx
import pytest

FRONTEND = os.environ.get("FRONTEND_URL", "http://127.0.0.1:3000")

CITATION = "sample_policy.md § Section 1: Home Water Damage Coverage"


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
    """One browser for the module, not one per test.

    Launching Chrome per test worked in isolation and errored intermittently
    when the whole suite ran - twenty-odd concurrent launches is enough to time
    one out. A suite that only passes when run alone is not a passing suite.
    """
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
    """A fresh page per test: the DOM is the thing under test, so it must not
    carry over, but the browser process can."""
    p = browser.new_page()
    p.goto(FRONTEND, wait_until="networkidle")
    p.wait_for_function("() => window.OmniCareVoice !== undefined", timeout=10_000)
    yield p
    p.close()


def feed(page, *messages: dict) -> None:
    """Deliver data-channel messages exactly as the worker publishes them."""
    page.evaluate("msgs => msgs.forEach(m => window.OmniCareVoice.handleData(m))",
                  list(messages))


def assistant_texts(page) -> list[str]:
    return page.evaluate(
        """() => [...document.querySelectorAll('.msg--assistant .bubble')]
                 .map(b => b.innerText.trim())"""
    )


# --------------------------------------------------------------- transcript

def test_the_assistants_spoken_words_land_in_the_thread(page) -> None:
    """The headline behaviour: a call now leaves a readable record."""
    before = len(assistant_texts(page))
    feed(
        page,
        {"type": "answer_delta", "text": "Sudden pipe bursts are covered "},
        {"type": "answer_delta", "text": "up to $25,000."},
        {"type": "state", "label": "listening", "kind": "ok"},
    )
    texts = assistant_texts(page)
    assert len(texts) == before + 1, "exactly one new assistant message"
    assert "Sudden pipe bursts are covered up to $25,000." in texts[-1]


def test_deltas_accumulate_rather_than_replacing(page) -> None:
    """A bold marker or a citation bracket can straddle two deltas, so the raw
    text is accumulated and re-rendered rather than appended as markup."""
    feed(
        page,
        {"type": "answer_delta", "text": "Your **Home Water "},
        {"type": "answer_delta", "text": "Damage** limit is $25,000."},
        {"type": "state", "label": "listening", "kind": "ok"},
    )
    # The `<p>`'s own innerHTML, not the bubble's: the raw text is accumulated in
    # a `data-raw` attribute on that element, so the bubble's innerHTML contains
    # the unrendered markers by design.
    html, text = page.evaluate(
        """() => {
            const p = [...document.querySelectorAll('.msg--assistant .bubble p')].pop();
            return [p.innerHTML, p.innerText];
        }"""
    )
    assert "<strong>Home Water Damage</strong>" in html
    assert "**" not in text, "the reader must never see the markers"


def test_citations_attach_to_the_answer_not_a_new_bubble(page) -> None:
    """`sources` was published as an `answer` carrying an empty string, so the
    browser opened a second, blank assistant bubble and the citations detached
    from the answer they belonged to."""
    before = len(assistant_texts(page))
    feed(
        page,
        {"type": "answer_delta", "text": "That is covered."},
        {"type": "sources", "sources": [CITATION]},
        {"type": "state", "label": "listening", "kind": "ok"},
    )
    assert len(assistant_texts(page)) == before + 1, "no second, empty bubble"
    cited = page.evaluate(
        """() => [...document.querySelectorAll('.msg--assistant .bubble')].pop()
                 .querySelector('.sources')?.innerText || ''"""
    )
    assert "Section 1: Home Water Damage Coverage" in cited


def test_a_confirmation_turn_leaves_no_empty_bubble(page) -> None:
    """A turn ending in a confirmation produces no tokens at all. The read-back
    is what the assistant said, so it belongs in the thread - and an empty card
    is worse than no card."""
    readback = "I'm about to file a Water Damage claim on policy P-O-L, one zero nine two."
    feed(page, {"type": "confirm", "readback": readback, "args": {}})

    texts = assistant_texts(page)
    assert all(t for t in texts), "no blank assistant bubble was left behind"
    assert readback in texts[-1], "the read-back is part of the conversation"
    assert page.evaluate("() => document.getElementById('confirm').hidden") is False


def test_the_callers_transcript_settles_from_partial_to_final(page) -> None:
    """Partial results are provisional and get corrected; they must update one
    bubble rather than appending a new one per interim result."""
    feed(page, {"type": "transcript_partial", "text": "is flood"})
    feed(page, {"type": "transcript_partial", "text": "is flood damage"})
    feed(page, {"type": "transcript_final", "text": "Is flood damage covered?"})

    user_texts = page.evaluate(
        """() => [...document.querySelectorAll('.msg--user .bubble')].map(b => b.innerText.trim())"""
    )
    assert user_texts[-1] == "Is flood damage covered?"
    assert "is flood" not in user_texts[:-1], "interim results must not pile up"


def test_model_markup_in_a_spoken_answer_cannot_inject(page) -> None:
    """The spoken answer reaches `innerHTML` by the same path as a typed one, so
    it inherits the same escaping. Asserted on the DOM, not on substrings."""
    feed(page, {"type": "answer_delta", "text": "<img src=x onerror=alert(1)>"})
    tags = page.evaluate(
        """() => [...[...document.querySelectorAll('.msg--assistant .bubble')].pop()
                 .querySelectorAll('*')].map(e => e.tagName.toLowerCase())"""
    )
    assert "img" not in tags


# --------------------------------------------------------------------- orb

def test_the_orb_draws_and_reports_its_state(page) -> None:
    """It must actually paint. A canvas that stays blank is the failure mode
    that looks like a working call with the camera off."""
    painted = page.evaluate(
        """() => {
            const c = document.createElement('canvas');
            c.width = c.height = 240;
            const orb = new window.OmniCareOrb(c);
            orb.setState('speaking');
            orb.level = 0.6;
            orb._draw();
            const data = c.getContext('2d').getImageData(0, 0, 240, 240).data;
            let lit = 0;
            for (let i = 3; i < data.length; i += 4) if (data[i] > 0) lit++;
            return lit;
        }"""
    )
    assert painted > 1000, "the orb drew nothing"


def test_the_orb_survives_having_no_audio_to_analyse(page) -> None:
    """A missing analyser costs the animation its reactivity and nothing more.
    The call must never fail because a visualiser could not start."""
    ok = page.evaluate(
        """() => {
            const c = document.createElement('canvas');
            c.width = c.height = 240;
            const orb = new window.OmniCareOrb(c);
            orb.attach(null, 'mic');           // no track
            for (const s of ['idle', 'listening', 'thinking', 'speaking']) {
                orb.setState(s);
                orb._draw();
            }
            orb.stop();
            return true;
        }"""
    )
    assert ok is True


def test_amplitude_moves_the_orb(page) -> None:
    """The size responds to the audio, which is the whole reason it is wired to
    an analyser rather than to a timer."""
    quiet, loud = page.evaluate(
        """() => {
            const measure = (level) => {
                const c = document.createElement('canvas');
                c.width = c.height = 240;
                const orb = new window.OmniCareOrb(c);
                orb.setState('speaking');
                orb.level = level;
                orb._draw();
                const d = c.getContext('2d').getImageData(0, 0, 240, 240).data;
                let lit = 0;
                for (let i = 3; i < d.length; i += 4) if (d[i] > 40) lit++;
                return lit;
            };
            return [measure(0), measure(1)];
        }"""
    )
    assert loud > quiet, f"a loud frame should cover more area ({loud} vs {quiet})"


# ------------------------------------------------------ the call as a surface

def viewport_fraction(page, selector: str) -> float:
    """How much of the viewport the element covers."""
    return page.evaluate(
        """sel => {
            const r = document.querySelector(sel).getBoundingClientRect();
            return (r.width * r.height) / (window.innerWidth * window.innerHeight);
        }""",
        selector,
    )


def test_the_call_takes_the_whole_screen(page) -> None:
    """On a call there is nothing to read and one thing to look at. A panel
    wedged under the transcript said "widget" for what is the entire foreground
    task."""
    page.evaluate("() => window.OmniCareVoice.openCall()")
    page.wait_for_timeout(150)

    assert page.evaluate("() => document.getElementById('voice-panel').hidden") is False
    assert viewport_fraction(page, "#voice-panel") > 0.95


def test_going_back_to_the_chat_does_not_end_the_call(page) -> None:
    """The call and the conversation are one thread, so hanging up to re-read an
    answer would be exactly backwards. Minimising leaves the room connected and
    only stops the drawing."""
    page.evaluate("() => window.OmniCareVoice.openCall()")
    page.evaluate("() => window.OmniCareVoice.minimiseCall()")
    page.wait_for_timeout(150)

    assert page.evaluate("() => document.getElementById('voice-panel').hidden") is True
    assert page.evaluate("() => document.getElementById('voice-return').hidden") is False


def test_a_minimised_call_can_be_returned_to(page) -> None:
    """Without the return control a minimised call is a call you cannot get
    back to."""
    page.evaluate("() => window.OmniCareVoice.openCall()")
    page.evaluate("() => window.OmniCareVoice.minimiseCall()")
    page.click("#voice-return")
    page.wait_for_timeout(150)

    assert page.evaluate("() => document.getElementById('voice-panel').hidden") is False
    assert page.evaluate("() => document.getElementById('voice-return').hidden") is True


def test_escape_goes_back_rather_than_hanging_up(page) -> None:
    """Escape is the dismiss gesture for an overlay. Dropping a live call on a
    stray keypress would be a nasty surprise."""
    page.evaluate("() => window.OmniCareVoice.openCall()")
    page.keyboard.press("Escape")
    page.wait_for_timeout(150)

    assert page.evaluate("() => document.getElementById('voice-panel').hidden") is True
    assert page.evaluate("() => document.getElementById('voice-return').hidden") is False


def test_the_chat_is_still_there_underneath(page) -> None:
    """Going back must return the caller to the conversation they left, with the
    transcript that the call has been writing into it."""
    feed(page, {"type": "answer_delta", "text": "Covered up to $25,000."},
         {"type": "state", "label": "listening", "kind": "ok"})
    page.evaluate("() => window.OmniCareVoice.openCall()")
    page.evaluate("() => window.OmniCareVoice.minimiseCall()")
    page.wait_for_timeout(150)

    assert "Covered up to $25,000." in assistant_texts(page)[-1]
    assert page.evaluate(
        "() => document.getElementById('composer').getBoundingClientRect().height"
    ) > 0, "the composer is usable again"


def test_the_orb_stops_drawing_when_it_is_not_on_screen(page) -> None:
    """A hidden canvas still costs a frame every 16 ms. `pause` keeps the audio
    graph - the agent's track has already been subscribed and will not fire
    again - while stopping the paint loop."""
    running = page.evaluate(
        """() => {
            const c = document.createElement('canvas');
            c.width = c.height = 200;
            const orb = new window.OmniCareOrb(c);
            // A stand-in for a live analyser: what matters is whether pause
            // keeps it and stop clears it, not how it was built.
            orb.analysers.agent = {node: null, buf: null};
            orb.start();
            const while_open = orb.raf !== null;
            orb.pause();
            const while_hidden = orb.raf !== null;
            const kept_audio = orb.analysers.agent !== null;
            orb.stop();
            const dropped_audio = orb.analysers.agent === null;
            return [while_open, while_hidden, kept_audio, dropped_audio];
        }"""
    )
    assert running == [True, False, True, True], (
        "pause must keep the analyser; stop must release it"
    )


def test_the_hidden_attribute_actually_hides(page) -> None:
    """A structural invariant, not a detail of one component.

    `[hidden] { display: none }` comes from the UA stylesheet, so any author
    rule that sets `display` on the same element outranks it. Two new
    components did exactly that, and the full-screen call overlay sat over the
    chat with `hidden` faithfully set - visible only as a click that landed on
    nothing.
    """
    offenders = page.evaluate(
        """() => [...document.querySelectorAll('[hidden]')]
                 .filter(el => getComputedStyle(el).display !== 'none')
                 .map(el => el.id || el.className)"""
    )
    assert offenders == [], f"hidden but still displayed: {offenders}"


# --------------------------------------------------------------- regressions

def test_the_orb_exists_before_any_track_can_arrive(page) -> None:
    """The orb used to be built after `await room.connect(...)`, while the
    `TrackSubscribed` handler was registered before it.

    So a track arriving during connect - which is what happens when the agent
    is already in the room, on a rejoin or when the worker got there first -
    found `orb` still null, skipped `attach`, and the agent's audio was never
    analysed. The orb then sat perfectly still for the whole call, while the
    assistant was speaking, which is the one moment it exists to show.
    """
    assert page.evaluate("() => window.OmniCareVoice.orb() !== null"), (
        "the orb must exist before a room can hand it a track"
    )


def test_remote_audio_elements_do_not_pile_up(page) -> None:
    """`track.attach()` appends an <audio> to the body and nothing removed it,
    so every call left one behind - and a stale element can still be playing
    the previous call's audio underneath the new one."""
    counts = page.evaluate(
        """() => {
            const V = window.OmniCareVoice;
            const before = document.querySelectorAll('audio').length;
            // Two calls' worth of attachments, then a hang-up.
            V._trackAudio(document.createElement('audio'));
            V._trackAudio(document.createElement('audio'));
            const during = document.querySelectorAll('audio').length;
            V._releaseAudio();
            return [before, during, document.querySelectorAll('audio').length];
        }"""
    )
    before, during, after = counts
    assert during == before + 2, "the elements were attached"
    assert after == before, f"{after - before} audio element(s) left behind"


# ------------------------------------------------------------- accessibility

def test_the_call_takes_focus_and_gives_it_back(page) -> None:
    """A full-screen overlay that does not move focus leaves a keyboard user
    tabbing through the chat behind it, invisibly."""
    page.focus("#mic")
    page.evaluate("() => window.OmniCareVoice.openCall()")
    page.wait_for_timeout(120)

    inside = page.evaluate(
        """() => document.getElementById('voice-panel')
                 .contains(document.activeElement)"""
    )
    assert inside, "focus stayed behind the overlay"

    page.evaluate("() => window.OmniCareVoice.minimiseCall()")
    page.wait_for_timeout(120)
    assert page.evaluate("() => document.activeElement.id") == "mic", (
        "focus was not returned to the control that opened the call"
    )


def test_the_chat_is_hidden_from_assistive_tech_during_a_call(page) -> None:
    """Otherwise a screen reader reads the conversation and the call surface as
    one continuous document."""
    page.evaluate("() => window.OmniCareVoice.openCall()")
    page.wait_for_timeout(120)
    hidden_during = page.evaluate(
        "() => document.querySelector('main').getAttribute('aria-hidden')"
    )
    page.evaluate("() => window.OmniCareVoice.minimiseCall()")
    page.wait_for_timeout(120)
    hidden_after = page.evaluate(
        "() => document.querySelector('main').getAttribute('aria-hidden')"
    )

    assert hidden_during == "true", "the chat is still exposed behind the call"
    assert hidden_after in (None, "false"), "the chat stayed hidden after going back"


def test_the_call_announces_itself_as_a_dialog(page) -> None:
    panel = page.evaluate(
        """() => {
            const el = document.getElementById('voice-panel');
            return [el.getAttribute('role'), el.getAttribute('aria-modal')];
        }"""
    )
    assert panel == ["dialog", "true"]
