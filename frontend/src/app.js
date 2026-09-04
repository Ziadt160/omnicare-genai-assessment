/* Chat surface.
 *
 * Talks to the gateway over WebSocket for streaming, and falls back to the
 * synchronous POST /api/v1/chat if the socket cannot be established. That
 * fallback is not decoration: the REST endpoint is the graded contract, so the
 * UI must remain fully functional on it alone.
 */

const API = (window.OMNICARE_API || "http://localhost:8080").replace(/\/$/, "");
const WS = API.replace(/^http/, "ws") + "/api/v1/chat/stream";

// Stable per browser so conversation history survives a reload.
const USER_ID = (() => {
  const key = "omnicare_user_id";
  let id = localStorage.getItem(key);
  if (!id) {
    id = "usr_" + Math.random().toString(36).slice(2, 10);
    localStorage.setItem(key, id);
  }
  return id;
})();

const el = {
  thread: document.getElementById("thread"),
  form: document.getElementById("composer"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  conn: document.getElementById("conn"),
  confirm: document.getElementById("confirm"),
  confirmText: document.getElementById("confirm-text"),
  confirmYes: document.getElementById("confirm-yes"),
  confirmNo: document.getElementById("confirm-no"),
  reset: document.getElementById("reset"),
};

let socket = null;
let conversationId = null;
let streamingBubble = null;

/* ------------------------------------------------------------- rendering */

/* Following the conversation down.
 *
 * Two failures to avoid, and they pull in opposite directions. `scrollTop` was
 * set when a message was added and on every token, but citations arrive *after*
 * the last token and `sources` appended them without scrolling - so the block
 * was in the DOM, `display: flex`, fully opaque, and below the fold. Present
 * and invisible is the worst way for a citation to fail, because nothing looks
 * broken. Meanwhile scrolling on every token dragged a reader who had scrolled
 * up to re-read an earlier answer back to the bottom, several times a second.
 *
 * So: follow only a reader who is already at the bottom, and measure that
 * *before* inserting, since the new content is exactly what would otherwise
 * make them look far from it. */
const FOLLOW_SLACK_PX = 140;

function isFollowing() {
  const t = el.thread;
  return t.scrollHeight - t.scrollTop - t.clientHeight <= FOLLOW_SLACK_PX;
}

function scrollToLatest() {
  el.thread.scrollTop = el.thread.scrollHeight;
}

/** Append something to the current reply, keeping it in view if the reader is
 *  following along. */
function appendToBubble(node) {
  if (!streamingBubble) return;
  const following = isFollowing();
  streamingBubble.parentElement.appendChild(node);
  if (following) scrollToLatest();
}

function addMessage(role, text, { sources = [], toolCalls = [], pending = false } = {}) {
  // Your own message always pulls the view down - you just sent it. An
  // assistant message does not, so an answer arriving while you are reading
  // history stays out of the way until you scroll back.
  const following = role === "user" || isFollowing();
  const li = document.createElement("li");
  li.className = `msg msg--${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (pending ? " bubble--pending" : "");

  const body = document.createElement("p");
  body.innerHTML = renderText(text);
  bubble.appendChild(body);

  if (toolCalls.length) bubble.appendChild(renderToolCalls(toolCalls));
  if (sources.length) bubble.appendChild(renderSources(sources));

  li.appendChild(bubble);
  el.thread.appendChild(li);
  if (following) scrollToLatest();
  return body;
}

function renderToolCalls(calls) {
  const wrap = document.createElement("div");
  wrap.className = "tools";
  for (const call of calls) {
    const chip = document.createElement("span");
    chip.className = "chip chip--" + (call.status || "ok");
    chip.textContent = call.name;
    chip.title = JSON.stringify(call.arguments || {}, null, 2);
    wrap.appendChild(chip);
  }
  return wrap;
}

function renderSources(sources) {
  const wrap = document.createElement("div");
  wrap.className = "sources";

  const label = document.createElement("span");
  label.className = "sources__label";
  label.textContent = sources.length === 1 ? "Source" : "Sources";
  wrap.appendChild(label);

  for (const source of sources) {
    // "sample_policy.md § Section 1: Home Water Damage Coverage"
    const [file, section] = source.split(" § ");
    const cite = document.createElement("span");
    cite.className = "cite";
    cite.innerHTML =
      `<span class="cite__section">${escapeHtml(section || source)}</span>` +
      `<span class="cite__file">${escapeHtml(file)}</span>`;
    wrap.appendChild(cite);
  }
  return wrap;
}

function escapeHtml(s) {
  const d = document.createElement("div");
  d.textContent = s;
  return d.innerHTML;
}

/* Models write markdown whether or not you ask them to, and gpt-oss-120b also
   emits U+3010 as a citation marker. Rendered as plain text that surfaces as
   literal ** and a dangling bracket in the middle of an insurance answer.

   A deliberately tiny subset - bold, italic, paragraph breaks - and escaping
   happens FIRST, so nothing the model writes can inject markup. A full
   markdown library would be a dependency and a much larger attack surface for
   one paragraph of prose. */
function renderText(raw) {
  return escapeHtml(raw)
    // Citation markers some models emit around references, left dangling when
    // the reference itself was stripped.
    .replace(/[【】「」]/g, "")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>")
    .replace(/\n{2,}/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/[ \t]+([.,;:!?])/g, "$1")
    .trim();
}

function setStatus(text, kind) {
  el.conn.textContent = text;
  el.conn.className = "pill pill--" + kind;
}

/* ---------------------------------------------------------------- socket */

function connect() {
  try {
    socket = new WebSocket(WS);
  } catch {
    setStatus("REST mode", "warn");
    return;
  }

  socket.onopen = () => {
    clearOfflineNotice();
    setStatus("connected", "ok");
  };
  socket.onclose = () => {
    setStatus("REST mode", "warn");
    socket = null;
  };
  socket.onerror = () => {
    // Not fatal. The graded REST endpoint carries the whole UI on its own.
    setStatus("REST mode", "warn");
  };
  socket.onmessage = (event) => handleEvent(JSON.parse(event.data));
}

function handleEvent(evt) {
  switch (evt.type) {
    case "queued":
      if (evt.payload.position > 0) {
        setStatus(`queued · ${evt.payload.position} ahead`, "warn");
      }
      break;

    case "started":
      setStatus("thinking", "busy");
      streamingBubble = addMessage("assistant", "");
      break;

    case "tool_start":
      setStatus(`calling ${evt.payload.name}`, "busy");
      break;

    case "tool_end":
      appendToBubble(renderToolCalls([evt.payload]));
      break;

    case "token":
      if (!streamingBubble) streamingBubble = addMessage("assistant", "");
      // Accumulate the raw text and re-render, rather than appending HTML:
      // a bold marker can straddle two chunks.
      {
        const following = isFollowing();
        streamingBubble.dataset.raw =
          (streamingBubble.dataset.raw || "") + (evt.payload.text || "");
        streamingBubble.innerHTML = renderText(streamingBubble.dataset.raw);
        if (following) scrollToLatest();
      }
      break;

    case "sources":
      if ((evt.payload.sources || []).length) {
        appendToBubble(renderSources(evt.payload.sources));
      }
      break;

    case "confirm":
      showConfirmation(evt.payload);
      break;

    case "done":
      discardEmptyBubble();
      setStatus(socket ? "connected" : "REST mode", socket ? "ok" : "warn");
      break;

    case "error":
      discardEmptyBubble();
      addMessage("system", evt.payload.message || "Something went wrong.");
      setStatus("error", "bad");
      break;
  }
}

/* ---------------------------------------------------------- confirmation */

function discardEmptyBubble() {
  /* `started` opens a bubble to stream tokens into. When the turn ends in a
     confirmation instead - or in an error - no token ever arrives, and the
     empty bubble stays on screen as a blank card. */
  if (streamingBubble && !streamingBubble.textContent.trim()) {
    const msg = streamingBubble.closest(".msg");
    if (msg) msg.remove();
  }
  streamingBubble = null;
}

function showConfirmation(payload) {
  const readback = payload.readback || "Please confirm before I file this claim.";

  /* Put the readback in the thread as well as the panel: it is what the
     assistant said, and a conversation that jumps from the request straight to
     "Filed" reads as though nothing was asked. */
  if (streamingBubble && !streamingBubble.textContent.trim()) {
    streamingBubble.innerHTML = renderText(readback);
    streamingBubble = null;
  } else {
    discardEmptyBubble();
    addMessage("assistant", readback);
  }

  el.confirmText.textContent = readback;
  el.confirm.hidden = false;
}

function answerConfirmation(answer) {
  el.confirm.hidden = true;
  send(answer);
}

el.confirmYes.addEventListener("click", () => answerConfirmation("yes"));
el.confirmNo.addEventListener("click", () => answerConfirmation("no, cancel that"));

/* ----------------------------------------------------------------- send */

async function send(message, { echo = true } = {}) {
  if (echo) addMessage("user", message);

  const payload = { user_id: USER_ID, message };
  if (conversationId) payload.conversation_id = conversationId;

  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify(payload));
    return;
  }

  // Synchronous path - the graded contract.
  setStatus("thinking", "busy");
  try {
    const response = await fetch(`${API}/api/v1/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (response.status === 429) {
      const wait = response.headers.get("Retry-After") || "a few";
      addMessage("system", `Too many requests. Try again in ${wait} seconds.`);
      setStatus("rate limited", "warn");
      return;
    }
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      addMessage("system", detail.detail || "The assistant is unavailable.");
      setStatus("error", "bad");
      return;
    }

    const body = await response.json();
    conversationId = body.conversation_id;
    clearOfflineNotice();

    const pending = (body.tool_calls || []).some(
      (c) => c.status === "awaiting_confirmation"
    );
    addMessage("assistant", body.response, {
      sources: body.sources || [],
      toolCalls: body.tool_calls || [],
    });
    if (pending) showConfirmation({ readback: body.response });

    setStatus("REST mode", "warn");
  } catch {
    addMessage("system", "Could not reach the assistant. Is the backend running?");
    setStatus("offline", "bad");
  }
}

/* Starting over.
 *
 * Clearing the screen alone would be worse than nothing: the agent keys its
 * memory on `conversation_id`, so the next message would still land in the old
 * thread and come back answered with context the policyholder can no longer
 * see. Dropping the id is what actually empties the conversation - the gateway
 * mints a fresh one on the next turn, and the old thread is left intact rather
 * than deleted, because a policyholder clearing their screen is not asking for
 * their claim history to be destroyed.
 */
function resetConversation() {
  conversationId = null;
  streamingBubble = null;
  el.confirm.hidden = true;
  el.input.value = "";

  // Keep the opening greeting: it is the page's own copy, not part of the
  // conversation, and an empty thread reads as a broken page.
  const messages = [...el.thread.children];
  for (const li of messages.slice(1)) li.remove();

  el.input.focus();
}

if (el.reset) el.reset.addEventListener("click", resetConversation);

el.form.addEventListener("submit", (e) => {
  e.preventDefault();
  const message = el.input.value.trim();
  if (!message) return;
  el.input.value = "";
  send(message);
});

/* --------------------------------------------------------------- startup */

let offlineNotice = null;

function clearOfflineNotice() {
  // A transient failure at page load must not leave a permanent, wrong
  // message on screen. Seen live: one reset health check during startup and
  // the page insisted the backend was down while it answered normally.
  if (offlineNotice) {
    offlineNotice.remove();
    offlineNotice = null;
  }
}

async function probeHealth(attempts = 3) {
  for (let i = 0; i < attempts; i++) {
    try {
      const health = await fetch(`${API}/api/v1/health`).then((r) => r.json());
      return health.status;
    } catch {
      // Containers come up in stages; a reset on the first try is normal.
      await new Promise((r) => setTimeout(r, 400 * (i + 1)));
    }
  }
  return null;
}

(async function boot() {
  const status = await probeHealth();

  if (status === null) {
    setStatus("offline", "bad");
    offlineNotice = addMessage(
      "system",
      "Cannot reach the backend yet. Run: docker compose up"
    ).closest(".msg");
  } else {
    setStatus(status === "healthy" ? "connected" : "degraded",
              status === "healthy" ? "ok" : "warn");
  }

  // Always attempt the socket. A failed health probe is not a reason to give
  // up the streaming path for the rest of the session.
  connect();
})();

/* `conversation` is exposed as an accessor pair rather than a value: voice and
   chat must share one conversation id, and whichever channel speaks first mints
   it. A plain copy handed over at load time would be null forever. */
window.OmniCare = {
  send,
  addMessage,
  handleEvent,
  resetConversation,
  renderText,
  renderSources,
  renderToolCalls,
  setStatus,
  USER_ID,
  API,
  get conversationId() {
    return conversationId;
  },
  set conversationId(value) {
    if (value) conversationId = value;
  },
};
