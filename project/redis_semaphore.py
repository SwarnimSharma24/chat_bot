
import os
import time
import uuid
import logging
import threading

import redis


logger = logging.getLogger("redis-semaphore")


# Redis connection configuration
REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://127.0.0.1:6379/0",
)

REDIS_KEY = "chatbot:llm:inference_slots"

# Distributed concurrency configuration
MAX_CONCURRENT_GENERATIONS = 1
LEASE_DURATION_MS = 60_000
HEARTBEAT_INTERVAL = 15
ACQUIRE_WAIT_TIMEOUT = 30
POLL_INTERVAL = 0.1


# Shared Redis client
redis_client = redis.Redis.from_url(
    REDIS_URL,
    decode_responses=True,
    socket_connect_timeout=2,
    socket_timeout=2,
)


# Atomically remove expired leases and acquire a slot.
ACQUIRE_SCRIPT = """
local key = KEYS[1]
local token = ARGV[1]
local max_slots = tonumber(ARGV[2])
local lease_ms = tonumber(ARGV[3])

local redis_time = redis.call('TIME')
local now = redis_time[1] * 1000
    + math.floor(redis_time[2] / 1000)

redis.call('ZREMRANGEBYSCORE', key, '-inf', now)

local active = redis.call('ZCARD', key)

if active < max_slots then
    redis.call('ZADD', key, now + lease_ms, token)
    redis.call('PEXPIRE', key, lease_ms * 2)
    return 1
end

return 0
"""


# Renew an existing lease only if it is still valid.
RENEW_SCRIPT = """
local key = KEYS[1]
local token = ARGV[1]
local lease_ms = tonumber(ARGV[2])

local redis_time = redis.call('TIME')
local now = redis_time[1] * 1000
    + math.floor(redis_time[2] / 1000)

local expiry = redis.call('ZSCORE', key, token)

if not expiry then
    return 0
end

if tonumber(expiry) <= now then
    redis.call('ZREM', key, token)
    return 0
end

redis.call('ZADD', key, now + lease_ms, token)
redis.call('PEXPIRE', key, lease_ms * 2)

return 1
"""


# Release only the lease owned by this request.
RELEASE_SCRIPT = """
local key = KEYS[1]
local token = ARGV[1]

return redis.call('ZREM', key, token)
"""


class SemaphoreAcquireTimeout(Exception):
    """Raised when no distributed slot becomes available."""
    pass


class SemaphoreLeaseLost(Exception):
    """Raised when the current request loses its lease."""
    pass


class RedisSemaphoreError(Exception):
    """Raised when Redis operations fail."""
    pass


class RedisDistributedSemaphore:

    def __init__(
        self,
        client=redis_client,
        key=REDIS_KEY,
        max_concurrent=MAX_CONCURRENT_GENERATIONS,
        lease_ms=LEASE_DURATION_MS,
        wait_timeout=ACQUIRE_WAIT_TIMEOUT,
    ):

        self.client = client
        self.key = key
        self.max_concurrent = max_concurrent
        self.lease_ms = lease_ms
        self.wait_timeout = wait_timeout

        self.acquire_script = self.client.register_script(
            ACQUIRE_SCRIPT
        )

        self.renew_script = self.client.register_script(
            RENEW_SCRIPT
        )

        self.release_script = self.client.register_script(
            RELEASE_SCRIPT
        )

    def acquire(self):

        token = str(uuid.uuid4())
        start_time = time.monotonic()

        while True:

            try:
                acquired = self.acquire_script(
                    keys=[self.key],
                    args=[
                        token,
                        self.max_concurrent,
                        self.lease_ms,
                    ],
                )

            except redis.RedisError as error:
                raise RedisSemaphoreError(
                    "Redis failed while acquiring an inference slot."
                ) from error

            if acquired:
                logger.info(
                    "Distributed slot acquired | token=%s",
                    token,
                )
                return token

            elapsed = time.monotonic() - start_time

            if elapsed >= self.wait_timeout:
                raise SemaphoreAcquireTimeout(
                    "LLM is busy. Please try again later."
                )

            time.sleep(
                min(
                    POLL_INTERVAL,
                    self.wait_timeout - elapsed,
                )
            )

    def renew(self, token):

        try:
            renewed = self.renew_script(
                keys=[self.key],
                args=[token, self.lease_ms],
            )

        except redis.RedisError as error:
            raise RedisSemaphoreError(
                "Redis failed while renewing the inference lease."
            ) from error

        if not renewed:
            raise SemaphoreLeaseLost(
                "The distributed inference lease has expired or been lost."
            )

        return True

    def release(self, token):

        try:
            released = self.release_script(
                keys=[self.key],
                args=[token],
            )

        except redis.RedisError as error:
            raise RedisSemaphoreError(
                "Redis failed while releasing the inference slot."
            ) from error

        if released:
            logger.info(
                "Distributed slot released | token=%s",
                token,
            )
        else:
            logger.warning(
                "Distributed slot already expired or released | token=%s",
                token,
            )

        return bool(released)

class LeaseHeartbeat:

    def __init__(
        self,
        semaphore,
        token,
        interval=HEARTBEAT_INTERVAL,
    ):
        self.semaphore = semaphore
        self.token = token
        self.interval = interval

        self._stop_event = threading.Event()
        self._lost_event = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="redis-lease-heartbeat",
            daemon=True,
        )

    def _run(self):

        while not self._stop_event.wait(self.interval):

            try:
                self.semaphore.renew(self.token)

                logger.debug(
                    "Lease heartbeat renewed | token=%s",
                    self.token,
                )

            except Exception as error:
                self._error = error
                self._lost_event.set()

                logger.exception(
                    "Lease heartbeat failed | token=%s",
                    self.token,
                )
                return

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        self._thread.join()

    def raise_if_lost(self):

        if not self._lost_event.is_set():
            return

        if isinstance(self._error, RedisSemaphoreError):
            raise RedisSemaphoreError(
                "Redis connection failed during inference heartbeat."
            ) from self._error

        raise SemaphoreLeaseLost(
            "Inference lease was lost during generation."
        ) from self._error

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()


# Shared semaphore instance for this Python process.
inference_semaphore = RedisDistributedSemaphore()
