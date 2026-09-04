-- Durable conversation state.
--
-- Two tables. LangGraph's checkpoint tables live in the `langgraph` schema and
-- are created by the saver's own setup() on first run.
--
-- Ownership: the gateway owns app.*, so history stays readable while the agent
-- is down. The agent owns langgraph.*. One instance, two schemas - production
-- would separate them, and the README says so.

CREATE SCHEMA IF NOT EXISTS app;
CREATE SCHEMA IF NOT EXISTS langgraph;

CREATE TABLE IF NOT EXISTS app.conversations (
  id          UUID PRIMARY KEY,
  user_id     TEXT        NOT NULL,
  title       TEXT,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS conversations_user_recent_idx
  ON app.conversations (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS app.messages (
  id               UUID PRIMARY KEY,
  conversation_id  UUID        NOT NULL
                     REFERENCES app.conversations(id) ON DELETE CASCADE,
  role             TEXT        NOT NULL
                     CHECK (role IN ('user', 'assistant', 'system')),
  content          TEXT        NOT NULL,
  sources          JSONB       NOT NULL DEFAULT '[]'::jsonb,
  tool_calls       JSONB       NOT NULL DEFAULT '[]'::jsonb,
  channel          TEXT        NOT NULL DEFAULT 'text',
  provider         TEXT,
  model            TEXT,
  latency_ms       INTEGER,
  -- Lets you jump from any message in chat history straight to its trace.
  trace_id         TEXT,
  created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS messages_conversation_idx
  ON app.messages (conversation_id, created_at);
