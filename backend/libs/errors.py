"""Domain exceptions.

Adapters translate driver errors into these. Nothing above the adapter layer
should ever see ``asyncpg.PostgresError``, ``redis.RedisError`` or a raw
``json.JSONDecodeError`` - that translation is what makes an adapter genuinely
swappable rather than a thin wrapper. See docs/adr/0005.
"""


class OmniCareError(Exception):
    """Base for every domain error."""


class ClaimNotFound(OmniCareError):
    def __init__(self, claim_id: str, known_ids: list[str] | None = None) -> None:
        self.claim_id = claim_id
        self.known_ids = known_ids or []
        super().__init__(f"No claim with id {claim_id!r}")


class DuplicateClaim(OmniCareError):
    def __init__(self, idempotency_key: str, existing_claim_id: str) -> None:
        self.idempotency_key = idempotency_key
        self.existing_claim_id = existing_claim_id
        super().__init__(f"Claim already filed as {existing_claim_id}")


class ConversationNotFound(OmniCareError):
    pass


class RateLimited(OmniCareError):
    def __init__(self, retry_after: int) -> None:
        self.retry_after = retry_after
        super().__init__(f"Rate limited; retry after {retry_after}s")


class QueueSaturated(OmniCareError):
    def __init__(self, depth: int, retry_after: int) -> None:
        self.depth = depth
        self.retry_after = retry_after
        super().__init__(f"Queue depth {depth} exceeds capacity")


class RunTimeout(OmniCareError):
    pass


class InjectionBlocked(OmniCareError):
    def __init__(self, rule_id: str) -> None:
        self.rule_id = rule_id
        super().__init__(f"Input blocked by rule {rule_id}")
