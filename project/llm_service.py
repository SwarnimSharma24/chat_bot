import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def generate_answer(prompt, model="llama3.2:1b"):
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
        raise RuntimeError(f"Ollama HTTP error: {error.code}") from error

    except URLError as error:
        raise RuntimeError(
            "Could not connect to Ollama. Make sure Ollama is running."
        ) from error
