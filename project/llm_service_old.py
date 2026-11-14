
import json
import threading
import logging
import time

from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


logger = logging.getLogger("llm-service")


# Maximum simultaneous LLM generations
MAX_CONCURRENT_GENERATIONS = 1

# Maximum time waiting for an inference slot
INFERENCE_WAIT_TIMEOUT = 30

# Shared semaphore for this Python process
_inference_semaphore = threading.BoundedSemaphore(
    MAX_CONCURRENT_GENERATIONS
)


class InferenceBusyError(Exception):
    """Raised when LLM inference capacity is unavailable."""
    pass


def generate_answer(prompt, model="llama3.2:1b"):

    thread_name = threading.current_thread().name

    # 1. Measure semaphore waiting time
    wait_start = time.perf_counter()

    logger.info(
        "Waiting for inference slot | thread=%s",
        thread_name
    )

    acquired = _inference_semaphore.acquire(
        timeout=INFERENCE_WAIT_TIMEOUT
    )

    if not acquired:
        wait_duration = round(
            time.perf_counter() - wait_start, 3
        )

        logger.warning(
            "Inference slot timeout | thread=%s | wait_duration=%ss",
            thread_name,
            wait_duration
        )

        raise InferenceBusyError(
            "LLM is busy. Please try again later."
        )

    wait_duration = round(
        time.perf_counter() - wait_start, 3
    )

    logger.info(
        "Inference slot acquired | thread=%s | wait_duration=%ss",
        thread_name,
        wait_duration
    )

    # 2. Measure actual Ollama generation time
    generation_start = time.perf_counter()

    try:
        url = "http://localhost:11434/api/chat"

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
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
                result = json.loads(response.read().decode("utf-8"))

            return result["message"]["content"]

        except HTTPError as error:
            raise RuntimeError(
                f"Ollama HTTP error: {error.code}"
            ) from error

        except URLError as error:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running."
            ) from error

    finally:
        # 3. Always release the semaphore
        generation_duration = round(
            time.perf_counter() - generation_start, 3
        )

        logger.info(
            "Releasing inference slot | thread=%s | generation_duration=%ss",
            thread_name,
            generation_duration
        )

        _inference_semaphore.release()
