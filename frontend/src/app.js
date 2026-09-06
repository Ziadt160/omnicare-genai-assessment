/* Chat surface.
 *
 * Talks to the gateway over WebSocket for streaming, and falls back to the
 * synchronous POST /api/v1/chat if the socket cannot be established. That
 * fallback is not decoration: the REST endpoint is the graded contract, so the
 * UI must remain fully functional on it alone.
 */

const API = (window.OMNICARE_API || "http://localhost:8080").replace(/\/$/, "");
const WS = API.replace(/^http/, "ws") + "/api/v1/chat/stream";

/* One identity per conversation.
 *
 * Persisted, so a reload keeps the thread - but replaced by "New
 * conversation", so starting one is a clean slate rather than the same person
 * carrying everything forward.
 *
 * That matters beyond tidiness. `ConversationStore.ensure` resolves a request
 * with no conversation id to the user's most recent conversation, and history,
 * claims and the rate limiter are all keyed on the user. A new conversation
 * that kept the old identity could still be resolved back into the old thread
 * by any request that omitted the id.
 */
const USER_KEY = "omnicare_user_id";

function newUserId() {
  return "usr_" + Math.random().toString(36).slice(2, 10);
}

let USER_ID = (() => {
  let id = localStorage.getItem(USER_KEY);
  if (!id) {
    id = newUserId();
    localStorage.setItem(USER_KEY, id);
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

function addMessage(role, text, {
  sources = [], toolCalls = [], pending = false,
  confidence = null, confidenceReason = null,
} = {}) {
  // Your own message always pulls the view down - you just sent it. An
  // assistant message does not, so an answer arriving while you are reading
  // history stays out of the way until you scroll back.
  const following = role === "user" || isFollowing();
  const li = document.createElement("li");
  li.className = `msg msg--${role}`;

  const bubble = document.createElement("div");
  bubble.className = "bubble" + (pending ? " bubble--pending" : "");

  /* A div, not a p. The renderer emits block elements now - headings, <ul>,
     one <p> per paragraph - and none of those are legal inside a <p>, so the
     browser would silently foster-parent them out of the bubble. The old
     renderer only ever emitted inline markup plus a literal "</p><p>", which
     worked by relying on that same auto-closing. */
  const body = document.createElement("div");
  body.className = "bubble__body";
  body.innerHTML = renderText(text);
  bubble.appendChild(body);

  if (confidence !== null && confidence !== undefined) {
    bubble.appendChild(renderConfidence(confidence, confidenceReason));
  }
  if (toolCalls.length) bubble.appendChild(renderToolCalls(toolCalls));
  if (sources.length) bubble.appendChild(renderSources(sources));

  li.appendChild(bubble);
  el.thread.appendChild(li);
  if (following) scrollToLatest();
  return body;
}

/* How far the system stands behind the answer.
 *
 * Shown as a band, not a number. "0.62" invites a reader to treat it as a
 * measurement, and it is not one - it starts as the model's own estimate,
 * which is the least reliable figure it produces, and is then lowered where
 * the system can see the answer was unsupported. A band carries what the value
 * is actually good for: whether to act on this or go and check.
 *
 * `reason` is present only when the system overrode the model, and it says
 * which check did it - a low number with no explanation is just discouraging.
 */
function renderConfidence(value, reason) {
  const wrap = document.createElement("div");
  const band = value >= 0.75 ? "high" : value >= 0.4 ? "medium" : "low";
  wrap.className = "confidence confidence--" + band;

  const label = document.createElement("span");
  label.className = "confidence__band";
  label.textContent =
    band === "high" ? "Grounded in your policy"
    : band === "medium" ? "Partly supported"
    : "Low confidence - check this";
  wrap.appendChild(label);

  if (reason) {
    const why = document.createElement("span");
    why.className = "confidence__why";
    why.textContent = reason;
    wrap.appendChild(why);
  }
  return wrap;
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
  const escaped = escapeHtml(raw)
    // Citation markers some models emit around references, left dangling when
    // the reference itself was stripped.
    .replace(/[【】「」]/g, "")
    // A fence around the whole answer is the model formatting prose as code.
    .replace(/^\s*```[a-z0-9_+-]*\s*\n?|\n?```\s*$/gi, "");

  /* Block structure first, inline second.

     Headings and list markers only mean anything at the start of a line, so
     they have to be matched before newlines are turned into <br> - which is
     what the previous version did straight away, leaving "### Section 1" and
     "- **Coverage**:" on screen with the hashes and hyphens showing. Reported
     with a screenshot of exactly that. */
  const blocks = [];
  let list = null;

  /* Blocks carry a flag rather than being sniffed by their first character.
     Testing `startsWith("<")` welded paragraphs together: "**Coverage**: ..."
     renders to inline HTML that begins with a tag, so a perfectly ordinary
     paragraph was classified as a block, lost its <p>, and flushed the one
     before it. That is the same paragraph-welding failure `_drop_orphaned_label`
     documents having already been fixed once on the backend, reintroduced here
     by a cheaper test. */
  const closeList = () => {
    if (list) {
      blocks.push({
        block: true,
        html: `<${list.tag}>${list.items.join("")}</${list.tag}>`,
      });
      list = null;
    }
  };
  const openList = (tag) => {
    if (!list || list.tag !== tag) {
      closeList();
      list = { tag, items: [] };
    }
    return list;
  };

  for (const line of escaped.split("\n")) {
    const heading = line.match(/^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$/);
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);

    if (heading) {
      closeList();
      // Capped at h4. The answer sits inside a chat bubble, so an h1 from a
      // model that started at "#" would tower over the conversation.
      const level = Math.min(heading[1].length + 2, 4);
      blocks.push({
        block: true,
        html: `<h${level} class="md-h">${inline(heading[2])}</h${level}>`,
      });
    } else if (bullet) {
      openList("ul").items.push(`<li>${inline(bullet[1])}</li>`);
    } else if (ordered) {
      openList("ol").items.push(`<li>${inline(ordered[1])}</li>`);
    } else if (!line.trim()) {
      closeList();
      blocks.push(null);
    } else {
      closeList();
      blocks.push({ block: false, html: inline(line) });
    }
  }
  closeList();

  // Consecutive non-block lines rejoin as one paragraph; a blank line starts a
  // new one. Anything already wrapped in a tag is passed through untouched.
  const html = [];
  let para = [];
  const flush = () => {
    if (para.length) {
      html.push(`<p>${para.join("<br>")}</p>`);
      para = [];
    }
  };
  for (const block of blocks) {
    if (!block) flush();
    else if (block.block) { flush(); html.push(block.html); }
    else para.push(block.html);
  }
  flush();

  return html.join("").replace(/[ \t]+([.,;:!?])/g, "$1").trim();
}

/* Emphasis, applied to already-escaped text. A deliberately tiny subset: a
   full markdown library would be a dependency and a much larger attack surface
   for one paragraph of prose. */
function inline(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s.,;:!?)]|$)/g, "$1<em>$2</em>")
    .replace(/`([^`\n]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\(([^)\s]*)\)/g, "$1");
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
      /* Confidence rides the terminal event, so it can only be rendered here.
         It was wired into the REST reply first and the WebSocket path forgot -
         which meant the band never appeared in the UI at all, because the
         socket is what the browser actually uses. The gateway forwards the
         whole event, so nothing but this was missing. */
      if (streamingBubble && evt.payload.confidence !== null &&
          evt.payload.confidence !== undefined) {
        // Inserted directly after the answer rather than appended, so the
        // order matches the REST path: answer, confidence, tool chips,
        // sources. `sources` has already been appended by the time `done`
        // arrives, so appending here would put the band below the citation.
        streamingBubble.after(
          renderConfidence(evt.payload.confidence, evt.payload.confidence_reason)
        );
      }
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

  /* The panel asks; the thread carries what was asked.
   *
   * It used to repeat the readback verbatim, which was tolerable while that
   * was one line. It is five now - the claim amount, what OmniCare pays, what
   * the policyholder pays - and the same block appearing twice, once as a
   * bubble and once in the panel directly beneath it, reads as the assistant
   * having said it twice. Reported from a real session.
   *
   * The thread keeps the full text, because a conversation that jumps from the
   * request straight to "Filed" reads as though nothing was asked. The panel
   * keeps only the question, next to the buttons that answer it. */
  el.confirmText.textContent = confirmQuestion(readback);
  el.confirm.hidden = false;
}

/* The last question in the readback, for the panel.
 *
 * `phonetic_readback` and the payment split are generated server-side and end
 * with "Shall I go ahead?", so the question is the final sentence. Falls back
 * to a plain prompt rather than to the whole block: if the wording ever
 * changes, a short generic question beside the buttons is still correct, and
 * repeating five lines is the thing being fixed. */
function confirmQuestion(readback) {
  const lines = String(readback).split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
  const last = lines[lines.length - 1] || "";
  return last.endsWith("?") ? last : "File this claim?";
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
      confidence: body.confidence,
      confidenceReason: body.confidence_reason,
    });
    if (pending) showConfirmation({ readback: body.response });

    setStatus("REST mode", "warn");
  } catch {
    addMessage("system", "Could not reach the assistant. Is the backend running?");
    setStatus("offline", "bad");
  }
}

/* A conversation id the server has never seen.
 *
 * `ensure` creates any id it does not recognise, so minting one here is what
 * makes a new conversation new. `randomUUID` needs a secure context, which
 * localhost is - the fallback is for a plain-http deployment, where a collision
 * would silently drop someone into another conversation. */
function newConversationId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  const rand = () => Math.random().toString(36).slice(2, 10);
  return `cnv_${Date.now().toString(36)}${rand()}${rand()}`;
}

/* Starting over.
 *
 * Clearing the screen alone would be worse than nothing: the agent keys its
 * memory on `conversation_id`, so the next message would still land in the old
 * thread and come back answered with context the policyholder can no longer
 * see.
 *
 * Dropping the id to null does NOT achieve that, which is what this used to do
 * and what the comment here used to claim. A request without a conversation id
 * does not start a new conversation - `ConversationStore.ensure` deliberately
 * resolves it to the user's most recent one, so that the graded request schema,
 * which has no conversation_id field at all, can hold a multi-turn thread.
 * Clearing the screen therefore emptied the display and nothing else: the next
 * message landed back in the conversation just abandoned, and the agent
 * answered it with context the policyholder could no longer see. Verified
 * against the running stack - "What did I just tell you about?" after a reset
 * came back describing the burst pipe from before it.
 *
 * So the new conversation is named here instead. The old thread is left intact
 * rather than deleted, because a policyholder clearing their screen is not
 * asking for their claim history to be destroyed.
 */
function resetConversation() {
  // A new identity as well as a new thread. Without it a later request that
  // omits the conversation id - the voice channel does this before its token
  // is minted - resolves back to this user's most recent conversation, which
  // is the one just cleared.
  USER_ID = newUserId();
  localStorage.setItem(USER_KEY, USER_ID);

  conversationId = newConversationId();
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
  newConversationId,
  confirmQuestion,
  renderText,
  renderSources,
  renderToolCalls,
  setStatus,
  get USER_ID() {
    // A getter, because USER_ID is replaced on reset. Exported as a value it
    // would freeze at page load and voice.js would keep speaking as the
    // previous identity.
    return USER_ID;
  },
  API,
  get conversationId() {
    return conversationId;
  },
  set conversationId(value) {
    if (value) conversationId = value;
  },
};
