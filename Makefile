.DEFAULT_GOAL := help
COMPOSE := docker compose

help:  ## Show this help
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

up:  ## Start everything (chat + voice)
	$(COMPOSE) up --build

up-chat:  ## Start without voice - the documented escape hatch
	$(COMPOSE) up --build postgres redis retrieval agent gateway frontend

up-obs:  ## Start with the Phoenix tracing UI on :6006
	$(COMPOSE) --profile obs up --build

up-groq:  ## Run against Groq (needs GROQ_API_KEY in .env)
	$(COMPOSE) --env-file .env --env-file .env.groq up -d --build

up-ollama:  ## Run against an Ollama on the host (no API key needed)
	$(COMPOSE) --env-file .env --env-file .env.local-ollama up -d --build

scale:  ## Demonstrate the consumer group distributing across 4 agent replicas
	$(COMPOSE) up --build --scale agent=4

down:  ## Stop and remove containers
	$(COMPOSE) down

clean:  ## Stop and delete volumes (resets claims, index and history)
	$(COMPOSE) down -v

test:  ## Every test layer, FakeLLM only, no network
	pytest tests/ -m "not live and not integration and not e2e" -q

test-all:  ## Including integration and e2e (needs Docker)
	pytest tests/ -m "not live" -q

eval:  ## The CI gate - deterministic, FakeLLM
	pytest evals/ -m "not live" -q

eval-live:  ## Real provider. Run once before submission; results go in the README.
	pytest evals/test_live.py::test_live_report -m live -q -s

docs:  ## Regenerate openapi.json and the Postman collection from the models
	python scripts/build_postman.py

seed:  ## Re-ingest the policy document into the vector store
	$(COMPOSE) exec retrieval python -m app.ingest

.PHONY: help up up-chat up-obs up-groq up-ollama scale down clean test test-all eval eval-live docs seed
