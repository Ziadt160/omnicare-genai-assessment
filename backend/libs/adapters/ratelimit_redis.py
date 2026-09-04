"""Token bucket in Redis, implemented as a Lua script.

The script is the point. A limiter that reads the count, decides, then writes
is not atomic: under concurrency two requests both read the same value, both
decide there is room, and both proceed. That is a suggestion, not a limit.
Lua runs the whole read-decide-write inside Redis as one operation.

This is *ingress* limiting - protecting the service from callers. The *egress*
limiter that protects us from the provider's own 429s lives in the agent, sized
to the free tier's RPM. Conflating the two is the classic mistake; they have
different windows, different keys, and different failure behaviour.
"""

from __future__ import annotations

import time

from redis.asyncio import Redis

from libs.errors import RateLimited

# KEYS[1] bucket   ARGV: capacity, refill_per_sec, now, cost
_SCRIPT = """
local bucket   = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill   = tonumber(ARGV[2])
local now      = tonumber(ARGV[3])
local cost     = tonumber(ARGV[4])

local state  = redis.call('HMGET', bucket, 'tokens', 'ts')
local tokens = tonumber(state[1])
local ts     = tonumber(state[2])

if tokens == nil then
  tokens = capacity
  ts = now
end

tokens = math.min(capacity, tokens + (now - ts) * refill)

local allowed = 0
if tokens >= cost then
  tokens = tokens - cost
  allowed = 1
end

redis.call('HMSET', bucket, 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', bucket, math.ceil(capacity / refill) + 1)

local retry_after = 0
if allowed == 0 then
  retry_after = math.ceil((cost - tokens) / refill)
end

return {allowed, retry_after}
"""


class RedisRateLimiter:
    def __init__(self, url: str, per_minute: int = 30, burst: int = 10) -> None:
        self._redis: Redis = Redis.from_url(url, decode_responses=True)
        self.capacity = float(max(burst, 1))
        self.refill_per_sec = per_minute / 60.0
        self._script = self._redis.register_script(_SCRIPT)

    async def check(self, scope: str, identity: str) -> None:
        """Consume one token, or raise ``RateLimited`` with a real Retry-After."""
        allowed, retry_after = await self._script(
            keys=[f"rl:{scope}:{identity}"],
            args=[self.capacity, self.refill_per_sec, time.time(), 1],
        )
        if not int(allowed):
            raise RateLimited(retry_after=max(1, int(retry_after)))

    async def ping(self) -> None:
        await self._redis.ping()

    async def aclose(self) -> None:
        await self._redis.aclose()
