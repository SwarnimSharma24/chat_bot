
import json
import threading
import logging
import time

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from redis_semaphore import (
    inference_semaphore,
    LeaseHeartbeat,
    SemaphoreAcquireTimeout,
    SemaphoreLeaseLost,
    RedisSemaphoreError,
)


logger = logging.getLogger("llm-service")


class InferenceBusyError(Exception):
    """Raised when LLM inference capacity is unavailable."""
    pass

class OllamaUnavailableError(Exception):
    """Raised when the Ollama service is unavailable."""
    pass

def generate_answer(prompt, model="llama3.2:1b"):

    thread_name = threading.current_thread().name

    # 1. Acquire a distributed Redis inference slot
    wait_start = time.perf_counter()

    logger.info(
        "Waiting for distributed inference slot | thread=%s",
        thread_name,
    )

    try:
        token = inference_semaphore.acquire()

    except (SemaphoreAcquireTimeout, RedisSemaphoreError) as error:

        wait_duration = round(
            time.perf_counter() - wait_start, 3
        )

        logger.warning(
            "Distributed inference slot unavailable | "
            "thread=%s | wait_duration=%ss",
            thread_name,
            wait_duration,
        )

        raise InferenceBusyError(
            "Inference capacity is temporarily unavailable."
        ) from error

    wait_duration = round(
        time.perf_counter() - wait_start, 3
    )

    logger.info(
        "Distributed inference slot acquired | "
        "thread=%s | wait_duration=%ss",
        thread_name,
        wait_duration,
    )

    # 2. Measure actual Ollama generation time
    generation_start = time.perf_counter()

    try:

        # Automatically renew the Redis lease during inference.
        with LeaseHeartbeat(
            inference_semaphore,
            token,
        ) as heartbeat:

            url = "http://localhost:11434/api/chat"

            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {"temperature": 0},
            }

            request = Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            try:
                with urlopen(request, timeout=180) as response:
                    result = json.loads(
                        response.read().decode("utf-8")
                    )

            except HTTPError as error:
                raise RuntimeError(
                    f"Ollama HTTP error: {error.code}"
                ) from error

            except URLError as error:
                raise OllamaUnavailableError(
                    "Ollama service is unavailable. "
                    "Make sure Ollama is running."
                ) from error

            # Never return an answer if the lease was lost.
            heartbeat.raise_if_lost()

            return result["message"]["content"]

    except (SemaphoreLeaseLost, RedisSemaphoreError) as error:

        logger.error(
            "Inference lease lost | thread=%s | error=%s",
            thread_name,
            error,
        )

        raise InferenceBusyError(
            "Inference capacity was lost. Please retry."
        ) from error

    finally:

        # 3. Always attempt to release the distributed slot.
        generation_duration = round(
            time.perf_counter() - generation_start, 3
        )

        logger.info(
            "Releasing distributed inference slot | "
            "thread=%s | generation_duration=%ss",
            thread_name,
            generation_duration,
        )

        try:
            inference_semaphore.release(token)

        except RedisSemaphoreError:
            logger.exception(
                "Failed to release Redis inference slot | token=%s",
                token,
            )
