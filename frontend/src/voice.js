/* Voice surface.
 *
 * The whole file is written around one rule from the spec: if LiveKit is
 * unreachable, the mic button disables itself with an explanation and chat is
 * completely unaffected. That is automatic graceful degradation rather than a
 * compose profile, and it is what protects a reviewer's first run on a machine
 * where WebRTC through Docker misbehaves.
 *
 * The live transcript is rendered into the DOM as it arrives over the LiveKit
 * data channel. Visual confirmation of what was heard costs nothing and
 * consumes no conversational turn - see docs/adr/0007.
 */

(function () {
  const API = window.OmniCare.API;
  const mic = document.getElementById("mic");
  const state = document.getElementById("voice-state");

  let room = null;
  let connected = false;
  let transcriptBubble = null;

  function setVoiceState(text, kind) {
    state.hidden = false;
    state.textContent = text;
    state.className = "pill pill--" + kind;
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

  async function start() {
    mic.disabled = true;
    setVoiceState("connecting", "busy");

    let credentials;
    try {
      credentials = await fetch(`${API}/api/v1/voice/token`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: window.OmniCare.USER_ID }),
      }).then((r) => r.json());
    } catch {
      disableVoice("Could not obtain a voice token — chat works normally.");
      return;
    }

    room = new LivekitClient.Room({ adaptiveStream: true, dynacast: true });

    room.on(LivekitClient.RoomEvent.ConnectionStateChanged, (s) => {
      if (s === "connected") setVoiceState("listening", "ok");
      if (s === "reconnecting") setVoiceState("reconnecting", "warn");
    });

    room.on(LivekitClient.RoomEvent.Disconnected, () => {
      connected = false;
      mic.disabled = false;
      mic.classList.remove("btn--live");
      setVoiceState("voice off", "idle");
    });

    /* The agent's spoken reply. */
    room.on(LivekitClient.RoomEvent.TrackSubscribed, (track) => {
      if (track.kind === "audio") document.body.appendChild(track.attach());
    });

    /* Transcripts, citations and queue position arrive on the data channel so
       the chat pane stays in sync with what is being spoken. */
    room.on(LivekitClient.RoomEvent.DataReceived, (payload) => {
      let msg;
      try {
        msg = JSON.parse(new TextDecoder().decode(payload));
      } catch {
        return;
      }

      if (msg.type === "transcript_partial") {
        if (!transcriptBubble) {
          transcriptBubble = window.OmniCare.addMessage("user", msg.text);
          transcriptBubble.parentElement.classList.add("bubble--pending");
        } else {
          transcriptBubble.textContent = msg.text;
        }
      } else if (msg.type === "transcript_final") {
        if (transcriptBubble) {
          transcriptBubble.textContent = msg.text;
          transcriptBubble.parentElement.classList.remove("bubble--pending");
          transcriptBubble = null;
        } else {
          window.OmniCare.addMessage("user", msg.text);
        }
      } else if (msg.type === "answer") {
        window.OmniCare.addMessage("assistant", msg.text, {
          sources: msg.sources || [],
          toolCalls: msg.tool_calls || [],
        });
      } else if (msg.type === "confirm") {
        // Spoken readback is handled by TTS; the panel mirrors it visually so
        // the policyholder can check the digits by eye as well as by ear.
        document.getElementById("confirm-text").textContent = msg.readback;
        document.getElementById("confirm").hidden = false;
      } else if (msg.type === "state") {
        setVoiceState(msg.label, msg.kind || "busy");
      }
    });

    try {
      await room.connect(credentials.url, credentials.token);
      await room.localParticipant.setMicrophoneEnabled(true);
      connected = true;
      mic.disabled = false;
      mic.classList.add("btn--live");
      mic.title = "Stop voice";
    } catch (err) {
      disableVoice("Voice could not connect — chat works normally.");
      window.OmniCare.addMessage(
        "system",
        "Voice is unavailable on this machine (WebRTC could not connect). " +
          "Chat is unaffected."
      );
    }
  }

  async function stop() {
    if (room) await room.disconnect();
    room = null;
  }

  mic.addEventListener("click", () => (connected ? stop() : start()));

  probe();
})();
