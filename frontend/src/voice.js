/* Voice surface.
 *
 * The whole file is written around one rule from the spec: if LiveKit is
 * unreachable, the mic button disables itself with an explanation and chat is
 * completely unaffected. That is automatic graceful degradation rather than a
 * compose profile, and it is what protects a reviewer's first run on a machine
 * where WebRTC through Docker misbehaves.
 *
 * Two things arrive over the LiveKit data channel and are rendered into the
 * same thread as typed messages:
 *
 *   - the caller's transcript, so they can see what was heard. Visual
 *     confirmation costs nothing and consumes no conversational turn - see
 *     docs/adr/0007.
 *   - the assistant's own words, streamed as they are spoken. A call used to
 *     leave no readable record of the answer: hang up and there was nothing to
 *     re-read, and a policyholder who misheard a deductible had no way to check
 *     it. Spoken and shown are not alternatives.
 *
 * The room carries the conversation id, so a call and a typed conversation are
 * one thread. Type, then press the mic, and the assistant already knows what
 * was said; a confirmation paused in chat can be resumed by voice.
 */

(function () {
  const API = window.OmniCare.API;
  const mic = document.getElementById("mic");
  const state = document.getElementById("voice-state");
  const panel = document.getElementById("voice-panel");
  const canvas = document.getElementById("voice-orb");
  const caption = document.getElementById("voice-caption");
  const phase = document.getElementById("voice-phase");
  const hangup = document.getElementById("voice-hangup");
  const meta = document.getElementById("voice-meta");
  const back = document.getElementById("voice-back");
  const returnBar = document.getElementById("voice-return");
  const elapsed = document.getElementById("voice-elapsed");

  let room = null;
  let connected = false;
  let transcriptBubble = null;
  let answerBubble = null;   // the assistant bubble currently being spoken into
  let startedAt = 0;
  let ticker = null;
  let restoreFocusTo = null;
  const remoteAudio = [];

  /* Built now, not after `room.connect()` resolves.

     The `TrackSubscribed` handler is registered before connect, so a track that
     arrives during it - which is what happens when the agent is already in the
     room - used to find `orb` still null and skip `attach` entirely. The
     agent's audio was then never analysed and the orb sat still for the whole
     call, while the assistant was speaking. Constructing it is cheap; `start()`
     is what costs frames, and that is still deferred. */
  const orb =
    canvas && window.OmniCareOrb ? new window.OmniCareOrb(canvas) : null;

  function setVoiceState(text, kind) {
    state.hidden = false;
    state.textContent = text;
    state.className = "pill pill--" + kind;
  }

  /* What a policyholder should see while a tool runs. `search_policy_documents`
     is the function's name; it belongs in the chip, where it is evidence of what
     ran, not in the status line, where it is just jargon mid-call. */
  const PHASE_LABELS = {
    search_policy_documents: "Checking your policy",
    get_claim_status: "Checking that claim",
    submit_claim: "Preparing the claim",
  };

  function phaseLabel(raw) {
    return PHASE_LABELS[raw] || raw;
  }

  /* Everything the current answer rests on, on the call surface itself.

     Citations replace rather than stack: they describe the whole turn, so a
     second `sources` event is a restatement, not an addition, and appending it
     shows the same section twice under one answer. Tool chips do stack - two
     tools running is two facts. */
  function showOnCall(node) {
    if (!meta) return;
    if (node.classList.contains("sources")) {
      const existing = meta.querySelector(".sources");
      if (existing) {
        existing.replaceWith(node);
        return;
      }
    }
    meta.appendChild(node);
  }

  function clearOnCall() {
    if (meta) meta.replaceChildren();
  }

  function setPhase(label, orbState) {
    if (phase) phase.textContent = phaseLabel(label);
    if (orb && orbState) orb.setState(orbState);
  }

  function disableVoice(reason) {
    mic.disabled = true;
    mic.title = reason;
    mic.setAttribute("aria-label", "Voice unavailable: " + reason);
    state.hidden = true;
  }

  /* Probe before offering the button at all. A mic button that fails on click
     is worse than one that was never enabled. */
  async function probe() {
    try {
      const response = await fetch(`${API}/api/v1/voice/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: window.OmniCare.USER_ID }),
      });
      if (!response.ok) throw new Error("token endpoint returned " + response.status);
      if (typeof LivekitClient === "undefined") {
        throw new Error("livekit-client did not load");
      }
      mic.disabled = false;
      mic.title = "Start voice";
      mic.setAttribute("aria-label", "Start voice");
      setVoiceState("voice ready", "idle");
    } catch (err) {
      disableVoice("Voice is unavailable — chat works normally. (" + err.message + ")");
    }
  }

  /* ------------------------------------------------------------ transcript */

  /* The assistant's reply, accumulated across deltas.

     Raw text is kept on the element and re-rendered rather than appended as
     HTML, for the same reason as the streaming chat path: a bold marker or a
     citation bracket can straddle two deltas, and appending markup would commit
     to a parse before the rest of the token arrived. */
  function appendAnswer(text) {
    if (!answerBubble) {
      answerBubble = window.OmniCare.addMessage("assistant", "");
      answerBubble.dataset.raw = "";
    }
    answerBubble.dataset.raw += text;
    answerBubble.innerHTML = window.OmniCare.renderText(answerBubble.dataset.raw);
  }

  /* Attach citations or tool chips to the reply they belong to. */
  function decorate(node) {
    if (!answerBubble) {
      answerBubble = window.OmniCare.addMessage("assistant", "");
      answerBubble.dataset.raw = "";
    }
    answerBubble.parentElement.appendChild(node);
  }

  function endTurn() {
    /* Drop a bubble that received nothing at all: a turn can end without ever
       producing a token, and an empty card is worse than no card.

       "Nothing" means no text *and* no chips or citations. Testing the
       paragraph alone was wrong in both directions - `decorate` attaches tool
       chips and sources as siblings of it, so a turn that retrieved a section
       but said nothing looked empty by that test and had its citations deleted
       along with the card. */
    if (answerBubble) {
      const bubble = answerBubble.parentElement;
      const bare =
        !answerBubble.textContent.trim() &&
        !(bubble && bubble.querySelector(".sources, .tools"));
      if (bare) {
        const msg = answerBubble.closest(".msg");
        if (msg) msg.remove();
      }
    }
    answerBubble = null;
  }

  function handleData(msg) {
    if (msg.type === "transcript_partial") {
      setPhase("Listening", "listening");
      if (caption) caption.textContent = msg.text;
      if (!transcriptBubble) {
        transcriptBubble = window.OmniCare.addMessage("user", msg.text);
        transcriptBubble.parentElement.classList.add("bubble--pending");
      } else {
        transcriptBubble.textContent = msg.text;
      }
    } else if (msg.type === "transcript_final") {
      // A new question: whatever the last answer rested on no longer applies,
      // and leaving it up would attribute this answer to a section that was
      // never consulted for it.
      clearOnCall();
      if (caption) caption.textContent = msg.text;
      if (transcriptBubble) {
        transcriptBubble.textContent = msg.text;
        transcriptBubble.parentElement.classList.remove("bubble--pending");
        transcriptBubble = null;
      } else if (msg.text) {
        window.OmniCare.addMessage("user", msg.text);
      }
      // The caller has stopped; the agent has the turn now.
      setPhase("Working", "thinking");
    } else if (msg.type === "answer_delta") {
      setPhase("Speaking", "speaking");
      if (caption) caption.textContent = "";
      appendAnswer(msg.text);
    } else if (msg.type === "sources") {
      // Both places: the call surface, so it is visible while the answer is
      // being spoken, and the thread, so it survives the call.
      decorate(window.OmniCare.renderSources(msg.sources || []));
      showOnCall(window.OmniCare.renderSources(msg.sources || []));
    } else if (msg.type === "tool") {
      decorate(window.OmniCare.renderToolCalls([msg]));
      showOnCall(window.OmniCare.renderToolCalls([msg]));
    } else if (msg.type === "confirm") {
      // Spoken readback is handled by TTS; the panel mirrors it visually so the
      // policyholder can check the digits by eye as well as by ear - and it goes
      // into the thread, because it is what the assistant said. A turn that ends
      // in a confirmation emits no tokens at all, so there is usually no bubble
      // yet; `appendAnswer` opens one.
      if (!answerBubble || !answerBubble.textContent.trim()) {
        appendAnswer(msg.readback || "");
      }
      endTurn();
      document.getElementById("confirm-text").textContent = msg.readback;
      document.getElementById("confirm").hidden = false;
      setPhase("Awaiting confirmation", "thinking");
    } else if (msg.type === "state") {
      setVoiceState(msg.label, msg.kind || "busy");
      if (msg.label === "listening") {
        endTurn();
        setPhase("Listening", "listening");
      } else {
        setPhase(msg.label, "thinking");
      }
    }
  }

  /* ------------------------------------------------------- the call surface */

  /* Three states, not two: the call is either closed, open full-screen, or
     running while the caller reads the chat. Collapsing the last two would mean
     hanging up to re-read an answer, which is the opposite of the point - the
     call and the conversation are one thread. */

  /* A full-screen overlay that does not move focus leaves a keyboard user
     tabbing through the chat behind it, invisibly, and a screen reader reading
     the conversation and the call as one continuous document. */
  function setBackgroundInert(on) {
    for (const el of [document.querySelector("main"),
                      document.querySelector(".masthead"),
                      document.getElementById("composer")]) {
      if (!el) continue;
      if (on) el.setAttribute("aria-hidden", "true");
      else el.removeAttribute("aria-hidden");
      // `inert` also takes them out of the tab order where it is supported;
      // aria-hidden alone would leave focusable controls reachable.
      el.inert = on;
    }
  }

  function restoreFocus() {
    // Only if focus is still inside the overlay: the caller may have clicked
    // into the chat, and yanking focus back would be worse than leaving it.
    const inside = panel && panel.contains(document.activeElement);
    if (restoreFocusTo && (inside || document.activeElement === document.body)) {
      restoreFocusTo.focus();
    }
    restoreFocusTo = null;
  }

  function openCall() {
    if (panel) panel.hidden = false;
    if (returnBar) returnBar.hidden = true;
    mic.title = "Return to the call";
    mic.setAttribute("aria-label", "Return to the call");
    if (panel) {
      restoreFocusTo =
        document.activeElement && document.activeElement !== document.body
          ? document.activeElement
          : mic;
      setBackgroundInert(true);
      // The panel itself, not the first button: "Back to chat" being read out
      // first tells a screen-reader user how to leave before telling them what
      // they arrived at.
      panel.focus();
    }
    if (orb) {
      // Measure after unhiding: the canvas is sized in `vmin` by the
      // stylesheet, so its box is 0 while the panel is hidden.
      orb.resize();
      // Only paint while there is something on screen to paint on; a hidden
      // canvas still costs a frame every 16 ms.
      orb.start();
    }
  }

  function minimiseCall() {
    setBackgroundInert(false);
    restoreFocus();
    if (panel) panel.hidden = true;
    if (returnBar) returnBar.hidden = false;
    if (orb) orb.pause();
    mic.title = "Return to the call";
    mic.setAttribute("aria-label", "Return to the call");
  }

  /* `track.attach()` hands back an <audio> element and appending it was the end
     of the story - so every call left one on the page, and a stale element can
     still be playing the previous call underneath the new one. */
  function trackAudio(el) {
    el.autoplay = true;
    document.body.appendChild(el);
    remoteAudio.push(el);
    return el;
  }

  function untrackAudio(el) {
    const i = remoteAudio.indexOf(el);
    if (i >= 0) remoteAudio.splice(i, 1);
    el.remove();
  }

  function releaseAudio() {
    while (remoteAudio.length) remoteAudio.pop().remove();
  }

  function closeCall() {
    if (panel) panel.hidden = true;
    if (returnBar) returnBar.hidden = true;
    if (orb) orb.stop();
    clearOnCall();
    releaseAudio();
    stopTicker();
    setBackgroundInert(false);
    restoreFocus();
  }

  function startTicker() {
    startedAt = Date.now();
    const tick = () => {
      const total = Math.floor((Date.now() - startedAt) / 1000);
      const mins = Math.floor(total / 60);
      const secs = String(total % 60).padStart(2, "0");
      if (elapsed) elapsed.textContent = `${mins}:${secs}`;
    };
    tick();
    ticker = setInterval(tick, 1000);
  }

  function stopTicker() {
    if (ticker) clearInterval(ticker);
    ticker = null;
    if (elapsed) elapsed.textContent = "0:00";
  }

  async function start() {
    mic.disabled = true;
    setVoiceState("connecting", "busy");

    let credentials;
    try {
      credentials = await fetch(`${API}/api/v1/voice/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: window.OmniCare.USER_ID,
          // Continue the conversation already on screen rather than opening a
          // second one the assistant has no memory of.
          conversation_id: window.OmniCare.conversationId || undefined,
        }),
      }).then((r) => r.json());
    } catch {
      disableVoice("Could not obtain a voice token — chat works normally.");
      return;
    }

    // Adopt the id the gateway minted, so typed turns after the call land on
    // the same thread the call used.
    window.OmniCare.conversationId = credentials.conversation_id;

    room = new LivekitClient.Room({ adaptiveStream: true, dynacast: true });

    room.on(LivekitClient.RoomEvent.ConnectionStateChanged, (s) => {
      if (s === "connected") setVoiceState("listening", "ok");
      if (s === "reconnecting") setVoiceState("reconnecting", "warn");
    });

    room.on(LivekitClient.RoomEvent.Disconnected, () => {
      connected = false;
      mic.disabled = false;
      mic.classList.remove("btn--live");
      mic.title = "Start voice";
      mic.setAttribute("aria-label", "Start voice");
      setVoiceState("voice off", "idle");
      closeCall();
      endTurn();
    });

    /* The agent's spoken reply: played by the attached element, and analysed by
       the orb so the animation is driven by real audio rather than a timer. */
    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
      if (track.kind !== "audio") return;
      trackAudio(track.attach());
      if (orb) orb.attach(track.mediaStreamTrack, "agent");
    });

    room.on(LivekitClient.RoomEvent.TrackUnsubscribed, (track) => {
      if (track.kind === "audio") track.detach().forEach(untrackAudio);
    });

    /* Transcripts, the answer, citations and queue position all arrive here so
       the chat pane stays in sync with what is being spoken. */
    room.on(LivekitClient.RoomEvent.DataReceived, (payload) => {
      let msg;
      try {
        msg = JSON.parse(new TextDecoder().decode(payload));
      } catch {
        return;
      }
      handleData(msg);
    });

    try {
      await room.connect(credentials.url, credentials.token);
      await room.localParticipant.setMicrophoneEnabled(true);
      connected = true;
      mic.disabled = false;
      mic.classList.add("btn--live");
      mic.title = "Stop voice";

      if (orb) {
        const pub = room.localParticipant.getTrackPublication(
          LivekitClient.Track.Source.Microphone
        );
        const micTrack = pub && pub.track && pub.track.mediaStreamTrack;
        if (micTrack) orb.attach(micTrack, "mic");
        orb.setState("listening");
      }
      openCall();
      startTicker();
      setPhase("Listening", "listening");
    } catch {
      // Let go of the Room and its listeners. Without this a failed attempt
      // left them attached and the next one overwrote the reference, so
      // every retry added another set of handlers to a room nobody could reach.
      try {
        await room.disconnect();
      } catch {
        /* it never connected */
      }
      room = null;
      releaseAudio();
      disableVoice("Voice could not connect — chat works normally.");
      window.OmniCare.addMessage(
        "system",
        "Voice is unavailable on this machine (WebRTC could not connect). " +
          "Chat is unaffected."
      );
    }
  }

  async function stop() {
    // `room = null` before awaiting: a disconnect that throws must still leave
    // the UI able to start a new call rather than stuck showing a live one.
    const leaving = room;
    room = null;
    connected = false;
    closeCall();
    if (leaving) {
      try {
        await leaving.disconnect();
      } catch {
        /* already gone */
      }
    }
  }

  /* While a call is running the mic reopens it rather than hanging up. Ending
     a call is deliberate and lives on one clearly-labelled control; a button
     that starts a call on one press and drops it on the next is how you lose a
     call you meant to return to. */
  mic.addEventListener("click", () => (connected ? openCall() : start()));
  if (back) back.addEventListener("click", minimiseCall);
  if (returnBar) returnBar.addEventListener("click", openCall);
  if (hangup) hangup.addEventListener("click", () => stop());

  /* Escape goes back to the chat rather than ending the call: it is the
     dismiss gesture for an overlay, and dropping a call on a stray keypress
     would be a nasty surprise. */
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && panel && !panel.hidden) minimiseCall();
  });

  /* Exposed for the browser tests. The data-channel message set is the entire
     contract between the voice worker and this file, and the only honest way to
     check that a spoken answer reaches the transcript is to feed the real
     messages to the real handler in a real browser. Driving it through an actual
     call would need a microphone and an SFU to assert on a DOM node. */
  window.OmniCareVoice = {
    handleData, endTurn, openCall, minimiseCall, closeCall,
    orb: () => orb,
    _trackAudio: trackAudio,
    _releaseAudio: releaseAudio,
  };

  probe();
})();
