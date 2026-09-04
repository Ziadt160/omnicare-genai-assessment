# Walkthrough

Every screenshot below was captured against the running stack by
[`scripts/capture_walkthrough.py`](../scripts/capture_walkthrough.py) — scripted
rather than hand-taken, so they can be regenerated after a UI change instead of
quietly going stale. Regenerate with:

```bash
docker compose up -d
python scripts/capture_walkthrough.py
```

Captured against **Groq `openai/gpt-oss-120b`** — these are real model answers,
real retrieval and real tool calls. With no key configured the system still
answers, from a keyless demo provider, and every reply says so; see
[Run it in two minutes](../README.md#run-it-in-two-minutes).

---

## 1. The chat surface

![The chat surface at rest](images/01-empty.png)

`CONNECTED` means the WebSocket is up; the page falls back to the synchronous
`POST /api/v1/chat` if it is not, because that endpoint is the graded contract
and the UI has to work on it alone. `VOICE READY` means the gateway minted a
LiveKit token — if it could not, the mic button disables itself with an
explanation and chat is unaffected.

---

## 2. A coverage question, answered with a citation

![Coverage answer with a section citation](images/02-coverage-citation.png)

The `search_policy_documents` chip shows the tool actually ran. Underneath,
**SOURCES** lists the section that was used, with the section title as the
heading and the file beneath it.

The citation is not reconstructed for display — it is carried as metadata on
the chunk from the moment the section is identified at ingest, through the
vector store payload and the tool result, into `sources` unchanged. Nothing
downstream can name the section differently from the index.

A citation the retrieval step did not return is stripped by the `ground` node
before the answer is rendered, which is what makes citation precision a
deterministic 1.00 rather than a hope.

---

## 3. An exclusion, stated plainly

![Exclusion stated plainly](images/03-exclusion.png)

`sample_policy.md` says gradual leaks and flood damage are **strictly
excluded**. The default failure of a small model asked "is water damage
covered?" is a confident yes, and that is the single most dangerous wrong
answer this system can give — so it is defended three times over: an explicit
instruction in the `search_policy_documents` docstring, the exclusion sitting
in the same chunk as the coverage limit so retrieval cannot return one without
the other, and gated eval cases `EV-04` and `EV-05` at recall 1.00.

---

## 4. Claim status, through a backend tool

![Claim lookup through the backend tool](images/04-claim-status.png)

`get_claim_status` reads `mock_claims.json`. The amounts stay JSON numbers on
write — `Decimal` serializes to a string by default, which would have silently
changed the shape of the file the brief tells us to append to.

---

## 5. An unknown claim recovers instead of dead-ending

![Unknown claim offers the real IDs](images/05-claim-recovery.png)

`CLM-8822` does not exist. Rather than "not found", the tool returns the
closest real IDs ranked by shared trailing digits — STT mangles the digits, not
the prefix. This is the most common way a voice assistant fails after a
mishearing, and it turns a dead end into a one-word correction.

---

## 6. Prompt injection is refused

![Prompt injection refused](images/06-injection-refused.png)

Note what is absent: **no tool chips and no sources.** The `guard` node screens
the message before any model call, so a blocked turn costs zero tokens and
zero quota.

The discrimination that matters, and that the suite asserts: *"what is the
status of CLM-8821"* is allowed; *"approve CLM-8821"* is blocked. A guardrail
that also blocks real policyholder questions has failed, not succeeded.

---

## 7. Filing a claim pauses first

![Irreversible write paused for confirmation](images/07-confirm-prompt.png)

`submit_claim` writes a permanent record, so the graph calls LangGraph's
`interrupt()` and stops. **Nothing has been written at this point.** The
response carries the pending call with status `awaiting_confirmation`.

The policy number is read back phonetically — "P-O-L, one zero nine two" — so a
policyholder can verify it by ear as well as by eye. That matters on the voice
channel, where the digits arrived through speech recognition.

What is confirmed is the **parsed arguments**, not the raw transcript. That one
gate catches transcription errors and model extraction errors together;
confirming a transcript would be noisier and would still miss a perfectly
transcribed `POL-1092` extracted as `POL-1029`.

---

## 8. Confirmed, and filed

![Claim filed after confirmation](images/08-claim-filed.png)

"yes" is a **separate HTTP request**. It resumes the paused graph from its
Postgres checkpoint, which means the follow-up can land on a different agent
replica than the one that paused — the reason the checkpointer is Postgres and
not memory. The confirmation ID is read back phonetically too.

```console
$ docker compose exec agent cat /data/claims/mock_claims.json
[
  { "claim_id": "CLM-8821", ... , "description": "" },
  { "claim_id": "CLM-9014", ... , "description": "" },
  {
    "claim_id": "CLM-9015",
    "policy_number": "POL-1092",
    "claim_type": "Water Damage",
    "status": "Submitted",
    "amount": 1200.0,
    "description": "File a water damage claim on POL-1092 for $1,200 - the washing machine hose burst."
  }
]
```

A retried `submit_claim` returns the original confirmation ID rather than
filing a second claim — the idempotency key is a hash of user, arguments and
conversation turn, so a network timeout after a successful write cannot produce
a duplicate insurance claim.

---

## 9. Declining writes nothing

![Declining writes nothing](images/09-declined.png)

The claims file is unchanged. `tests/e2e/test_stack.py::test_declining_writes_nothing`
asserts this against the live stack.

---

## Voice: end to end

Speech is local. Piper — a 63 MB ONNX voice baked into the image, CPU at about
five times realtime — so the voice channel needs no account, no network and no
quota, under the same zero-cost constraint as everything else. Groq synthesis
is available behind `VOICE_TTS_PROVIDER=groq` if you prefer the voice and have
accepted its terms.

The worker is ears and a mouth and nothing more. `OmniCareLLM` presents the
agent to LiveKit as an `llm.LLM`, so a spoken turn goes onto the same
`jobs:chat` queue as a typed one and inherits injection screening, the bounded
loop, `interrupt()` before a write and citation grounding by construction.
There is no second prompt and no second tool loop to keep in sync.

```console
$ docker compose logs voice
INFO omnicare.voice.tts: Piper voice loaded from en_US-amy-low.onnx at 16000 Hz
INFO omnicare.voice: speech synthesis ready (piper): 22060 bytes at 16000 Hz
INFO livekit.agents: registered worker
INFO livekit.agents: received job request
INFO omnicare.voice: serving room omnicare-vg2 for vg2

$ docker compose logs livekit | grep "mediaTrack published"
... "participant": "vg2"                     # the caller
... "participant": "agent-AJ_cJsUqmTgJwti"   # the assistant, speaking
```

That last line is the whole thing: the assistant published a synthesised audio
track into the room.

Two details the graph adds for voice and not for chat. The identifier read-back
is **generated in code**, not prompted — asking a 7B for a fixed format
produced "CLM-eight eight twenty-one" about half the time, which is exactly the
ambiguity a read-back exists to remove. And a spoken filler goes out the moment
a tool starts, because several seconds of silence on a call reads as a dropped
connection rather than as thinking.

## Voice: what a call looks like

![The call, full screen](images/10-voice-call.png)

A call takes the whole screen. There is nothing to read on one and exactly one
thing to look at, and a 190px orb wedged under the transcript said "widget" for
what is the entire foreground task.

The orb is driven by an `AnalyserNode` on the real audio — the microphone track
while the caller speaks, the agent's subscribed track while it answers — not by
a timer. That is the point of it: a decorative animation looks identical whether
or not media is flowing, and media failing silently, with the room connected and
no sound, is exactly how WebRTC through Docker goes wrong. **If the orb moves,
media is flowing.**

![The three orb states, in dark mode](images/11-voice-states-dark.png)

Working has no amplitude to show, so it gets a sweep rather than a pulse.
Colours come from the same custom properties as the rest of the page, so the orb
follows the theme instead of carrying a second palette.

### Going back does not hang up

![Back in the chat, call still running](images/11-voice-minimised.png)

The call has three states, not two: closed, full screen, or **running while the
caller reads the chat**. Collapsing the last two would mean hanging up to re-read
an answer, which is precisely backwards — the call and the conversation are one
thread. Minimising stops the drawing and keeps the audio graph; the pill above
the composer is how you get back, because a minimised call you cannot return to
is a lost call. `Escape` minimises rather than ends, for the same reason.

While a call is live the mic button reopens it instead of hanging up. Ending is
deliberate and lives on one clearly-labelled control: a button that starts a call
on one press and drops it on the next is how you lose a call you meant to keep.

**The answer is in the thread.** Every word the assistant speaks is streamed to
the transcript as it is said — markdown rendered, citation attached — next to
the caller's own words. Before this only the caller's half was shown: the reply
existed as audio and nothing else, so hanging up left nothing to re-read and a
policyholder who misheard a deductible had no way to check it.

**It is one conversation, not two.** The room name carries the conversation id,
so the call continues whatever was already typed. The agent keys its memory on
`conversation_id`; without this the caller would have to introduce themselves
again, and a confirmation paused in chat could not be resumed by voice.

![The call on a phone](images/12-voice-call-mobile.png)

### What a real call found

The voice path was reported as "nothing responds". It was driven end to end with
Piper synthesising an utterance and Chrome playing it back as a fake microphone,
which is the only way to get a repeatable call without a person in the room.
Three separate faults, none of which a unit test would have shown:

**The voice worker was given 30 s while the agent was allowed 280 s.**
`.env.local-ollama` raises the model, agent and gateway budgets for a local
model and never raised the voice one, so a turn that the agent answered
correctly in 45 s had its reply thrown away and the caller heard "Sorry, I
wasn't able to complete that."

**Whisper transcribes silence as words.** Not an empty string - confident,
ordinary English. A quiet room produced a steady stream of "Bye.", "Mm.",
"Thank you." and "Thanks for watching!", and every one of them was a full agent
turn: a graph run, a model call and a spoken reply to something nobody said.
They are filtered on the whole utterance, never on length or confidence,
because the confirmation vocabulary is one word long and swallowing "no" would
make an irreversible write unconfirmable by voice.

**The greeting never played**, because the session was ending before it ran.

Afterwards the same test produced a clean call:

```
transcript_final  " ...my jewellery was stolen from my home ... and submit claim for that."
confirm           "I'm about to file a Personal Property claim on policy
                   P-O-L, one two three four five for $5000. Shall I go ahead?"
```

### Letting the caller finish

`livekit-agents` ends a turn after **0.5 s** of silence by default. That is tuned
for quick exchanges and is wrong here: a policyholder reading an identifier off a
letter — "claim … C-L-M … eight eight two one" — pauses mid-token, and being cut
off there produces half a transcript the agent then has to ask about again.
Waiting is cheaper than re-asking, so the floor is **1.8 s** with a 6 s ceiling,
both configurable.

Fixed rather than dynamic endpointing: dynamic adapts its delay from the history
of the call, and the turns that most need patience are the rare ones — someone
spelling out a policy number — so a moving average trained on short questions is
at its least patient exactly when it should be most.

---

## Voice: the WebRTC gate

Voice is the riskiest part of the build, because WebRTC through Docker Desktop
on Windows fails in a way that looks like success — the room connects and there
is simply no audio. So it has its own gate, run before any voice code:

```bash
docker compose up livekit
python infra/spike/mint_token.py
```

```
19:40:12.311  connecting to ws://localhost:7880 ...
19:40:12.658  signal connected (ws 7880 ok)
19:40:14.180  connection state: connected
19:40:14.180  room joined: spike
19:40:15.220  synthetic 440 Hz tone published (no permission needed)
19:40:17.177  ICE CONNECTED over udp / prflx
19:40:17.178  GATE PASSED - WebRTC media works through Docker on this machine.
```

**Gate passed on this machine**, over UDP, without needing the TCP fallback.

The page publishes a synthetic 440 Hz tone rather than a microphone track, so
the gate runs headlessly and in CI. ICE cannot tell the difference — the
negotiation being tested is identical, and it is the negotiation that fails
through Docker, not the audio source.

Two real bugs were found running this: LiveKit 1.8 rejects API secrets under 32
characters (it logs an ERROR and starts anyway, so the failure would have
surfaced later as tokens that will not validate), and `infra/spike/token.py`
shadowed the standard library's `token` module, breaking `python -m http.server`
in that directory. Hence `mint_token.py`.

---

## Scaling

```bash
docker compose up --scale agent=4
```

The Redis Streams consumer group distributes work across all four replicas with
no configuration change. Eight concurrent requests, measured:

```
req1 200 3.34s   req5 200 6.37s   req8 200 8.09s   req7 200 13.04s
req3 200 6.39s   req6 200 6.38s   req2 200 13.08s  req4 200 13.14s
```

Two waves of four clearing in sequence, which is what a working queue looks
like. Give the replicas about twenty seconds to warm before load — four of them
running the checkpointer's `setup()` DDL at once will make the first requests
time out.

---

## Observability

```bash
docker compose --profile obs up
```

Phoenix on `:6006`. Instrumented with OpenTelemetry and OpenInference
conventions rather than a vendor SDK, so the backend is one variable —
`OTEL_EXPORTER_OTLP_ENDPOINT`. Unset, every tracing call is a no-op and the
system runs unchanged.

Each response carries its `trace_id`, also returned as `X-Trace-Id` and stored
on the `messages` row:

```console
$ curl -si -X POST localhost:8080/api/v1/chat -d '{"user_id":"u","message":"..."}' | grep -i trace
x-trace-id: ff064ecf9abee6cfc5a18bf1f027059d
```

That link is what makes tracing useful during a demo — you can go from any
message in chat history straight to its span tree — rather than a screenshot of
a dashboard.

---

## What the tests cover

```bash
make test        # 292 in-process, no container, no network
pytest tests/e2e -m e2e   # 57 against the running stack
make eval        # the behavioural gate
```

| Layer | Count |
|---|---:|
| `tests/unit` | 214 |
| `tests/contract` | 47 |
| `tests/e2e` | 57 |
| `evals` | 31 |

All six eval gates green: citation precision 1.00 · exclusion recall 1.00 ·
injection block rate 1.00 · unconfirmed writes 1.00 · tool selection 1.00
(gate 0.90) · tool-argument match 1.00 (gate 0.95).

The e2e suite passes against **both** vector backends — in-memory and Qdrant —
which is what makes the `VectorStore` port a demonstration rather than an
assertion.
