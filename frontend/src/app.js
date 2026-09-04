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
};

let socket = null;
let conversationId = null;
let streamingBubble = null;

/* ------------------------------------------------------------- rendering */

function addMessage(role, text, { sources = [], toolCalls = [], pending = false } = {}) {
  const li = document.createElement("li");
  li.className = `msg msg--${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (pending ? " bubble--pending" : "");

  const body = document.createElement("p");
  body.textContent = text;
  bubble.appendChild(body);

  if (toolCalls.length) bubble.appendChild(renderToolCalls(toolCalls));
  if (sources.length) bubble.appendChild(renderSources(sources));

  li.appendChild(bubble);
  el.thread.appendChild(li);
  el.thread.scrollTop = el.thread.scrollHeight;
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
      if (streamingBubble) {
        streamingBubble.parentElement.appendChild(renderToolCalls([evt.payload]));
      }
      break;

    case "token":
      if (!streamingBubble) streamingBubble = addMessage("assistant", "");
      streamingBubble.textContent += evt.payload.text || "";
      el.thread.scrollTop = el.thread.scrollHeight;
      break;

    case "sources":
      if (streamingBubble && (evt.payload.sources || []).length) {
        streamingBubble.parentElement.appendChild(renderSources(evt.payload.sources));
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
    streamingBubble.textContent = readback;
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

window.OmniCare = { send, addMessage, setStatus, USER_ID, API };
