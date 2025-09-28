from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import requests
from openai import OpenAI

DEFAULT_TONGYI_MODEL = "qwen-plus"
TONGYI_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PARAMETERS_FILENAME = "params.json"

os.environ["TONGYI_API_KEY"] = "sk-e2998a5a85ca499885f9476278ac02db"


class ParameterExtractionError(RuntimeError):
    """Raised when the parameter extraction pipeline fails."""



def call_tongyi(
    prompt: str,
    *,
    model: str = DEFAULT_TONGYI_MODEL,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
) -> str:
    """Call Tongyi Qianwen (DashScope) chat completion API and return the text."""

    key = api_key or os.getenv("TONGYI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise EnvironmentError(
            "TONGYI_API_KEY (or DASHSCOPE_API_KEY) is not set; cannot call Tongyi API."
        )

    client = OpenAI(
        api_key=key,
        base_url=TONGYI_ENDPOINT,
    )

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )

    body = completion.model_dump()

    choices = body.get("choices")
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        message = first_choice.get("message") if isinstance(first_choice, dict) else None
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content

    raise ParameterExtractionError(
        f"Tongyi response missing content: {json.dumps(body, ensure_ascii=False)}"
    )


def load_json(path: str | Path) -> Dict:
    with open(Path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

