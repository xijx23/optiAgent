from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import requests
from utils import call_tongyi
from openai import OpenAI

DEFAULT_MODEL = "glm-4-plus"
CHATGLM_ENDPOINT = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
DEFAULT_TONGYI_MODEL = "qwen-plus"
TONGYI_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1"
PARAMETERS_FILENAME = "params.json"

os.environ["TONGYI_API_KEY"] = "sk-e2998a5a85ca499885f9476278ac02db" 

class ParameterExtractionError(RuntimeError):
    """Raised when the parameter extraction pipeline fails."""

@dataclass
class ParameterExtractionResult:
    parameters: Dict[str, Dict[str, object]]
    raw_response: str
    prompt: str

def _build_prompt(description: str) -> str:
    return (
        "You are an optimization modeling expert. Given the natural language "
        "description of an optimization problem, extract every known parameter "
        "and express them in a strict JSON object.\n\n"
        "Description:\n-----\n"
        f"{description}\n"
        "-----\n\n"
        "Rules:\n"
        "1. Each top-level key is the parameter name in CamelCase.\n"
        "2. Each value is an object with the fields 'definition', 'shape', 'type', 'value'.\n"
        "3. 'shape' must be a Python-style list literal: [] for scalar, [N], [N, M], etc.\n"
        "4. 'type' must be one of 'int', 'float', 'binary'.\n"
        "5. If a numeric value is explicitly stated in the description, place it in 'value' "
        "using numbers or lists; otherwise use null.\n"
        "6. Do not include commentary before or after the JSON object.\n\n"
        "Example output:\n"
        "{\n"
        "  \"NumberOfFactories\": {\n"
        "    \"definition\": \"How many factories can produce the goods\",\n"
        "    \"shape\": \"[]\",\n"
        "    \"type\": \"int\",\n"
        "    \"value\": 3\n"
        "  },\n"
        "  \"DemandPerRegion\": {\n"
        "    \"definition\": \"Demand for each served region\",\n"
        "    \"shape\": \"[R]\",\n"
        "    \"type\": \"float\",\n"
        "    \"value\": [1200.0, 950.0, 640.0]\n"
        "  }\n"
        "}\n"
    )



def _extract_json_block(text: str) -> Dict[str, Dict[str, object]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ParameterExtractionError("No JSON object found in model response.")
    snippet = text[start : end + 1]
    snippet = snippet.replace("\\n", "\n")
    return json.loads(snippet)

def _shape_to_list(shape_value: object) -> list:
    if isinstance(shape_value, list):
        return shape_value
    if isinstance(shape_value, str):
        cleaned = shape_value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            inner = cleaned[1:-1].strip()
            if not inner:
                return []
            parts = [part.strip() for part in inner.split(",")]
            result = []
            for part in parts:
                if part.isdigit():
                    result.append(int(part))
                else:
                    result.append(part)
            return result
    raise ParameterExtractionError(f"Cannot interpret shape value: {shape_value!r}")

def extract_parameters(description: str, *, model: str = DEFAULT_MODEL) -> ParameterExtractionResult:
    prompt = _build_prompt(description)
    raw = call_tongyi(prompt, model=model)
    parsed = _extract_json_block(raw)

    canonical: Dict[str, Dict[str, object]] = {}
    for name, payload in parsed.items():
        if not isinstance(payload, dict):
            raise ParameterExtractionError(f"Invalid parameter payload for {name!r}: {payload!r}")
        definition = payload.get("definition", "").strip()
        param_type = payload.get("type", "").strip().lower()
        shape_raw = payload.get("shape", "[]")
        value = payload.get("value")

        if not definition:
            raise ParameterExtractionError(f"Parameter {name} is missing a definition.")
        if param_type not in {"int", "float", "binary"}:
            raise ParameterExtractionError(
                f"Parameter {name} has unsupported type: {param_type!r}"
            )

        canonical[name] = {
            "definition": definition,
            "type": param_type,
            "shape": _shape_to_list(shape_raw),
            "value": value,
        }

    return ParameterExtractionResult(parameters=canonical, raw_response=raw, prompt=prompt)

def extract_and_store_parameters(
    problem_dir: Path | str,
    *,
    model: str = DEFAULT_MODEL,
    description: Optional[str] = None,
    output_filename: str = PARAMETERS_FILENAME,
) -> ParameterExtractionResult:
    root = Path(problem_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Problem directory not found: {root}")

    desc_text = (
        description
        if description is not None
        else (root / "desc.txt").read_text(encoding="utf-8")
    )
    desc_text = desc_text.strip()
    if not desc_text:
        raise ValueError("Problem description cannot be empty.")

    result = extract_parameters(desc_text, model=model)

    output_path = root/ "problems/demo-ed" / output_filename
    output_path.write_text(
        json.dumps(result.parameters, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Also persist the prompt and raw model response for traceability during debugging.
    debug_payload = {
        "prompt": result.prompt,
        "response": result.raw_response,
    }
    (root / "problems/demo-ed" / "params_extraction_log.json").write_text(
        json.dumps(debug_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return result



__all__ = [
    "extract_parameters",
    "extract_and_store_parameters",
    "ParameterExtractionResult",
    "ParameterExtractionError",
]

