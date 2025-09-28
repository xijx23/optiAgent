from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

from utils import DEFAULT_TONGYI_MODEL, call_tongyi


class ConstraintExtractionError(RuntimeError):
    """Raised when the constraint extraction pipeline fails."""


@dataclass
class ConstraintExtractionResult:
    constraints: List[Dict[str, Optional[str]]]
    raw_response: str
    prompt: str


CONSTRAINT_PROMPT_TEMPLATE = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

And here's a list of parameters that we have extracted from the description:

{params}


Your task is to identify and extract constraints from the description. The constraints are the conditions that must be satisfied by the variables. Please generate the output in the following python list format:

[
    Constraint 1,
    Constraint 2,
    ...
]

for example:
    
[
    "Sum of weights of all items taken should not exceed the maximum weight capacity of the knapsack", 
    "The number of items taken should not exceed the maximum number of items allowed"
]

- Put all the constraints in a single python list.
- Do not generate anything after the python list.
- Include implicit non-negativity constraints if necessary.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


def _build_prompt(description: str, parameters: Dict[str, Dict[str, object]]) -> str:
    params_json = json.dumps(parameters, indent=4, ensure_ascii=False) if parameters else "{}"
    return CONSTRAINT_PROMPT_TEMPLATE.format(description=description, params=params_json)


def _parse_constraints(text: str) -> List[str]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ConstraintExtractionError("No JSON array found in model response.")
    snippet = text[start : end + 1]
    try:
        data = json.loads(snippet)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise ConstraintExtractionError("Failed to decode constraint list from model response.") from exc
    if not isinstance(data, list):
        raise ConstraintExtractionError("Model response is not a JSON array.")

    constraints: List[str] = []
    for item in data:
        if not isinstance(item, str):
            raise ConstraintExtractionError("Each constraint must be a string description.")
        cleaned = item.strip()
        if cleaned:
            constraints.append(cleaned)
    if not constraints:
        raise ConstraintExtractionError("No constraints were extracted from the model response.")
    return constraints


def get_constraints(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    *,
    model: str = DEFAULT_TONGYI_MODEL,
    temperature: float = 0.2,
) -> ConstraintExtractionResult:
    if not description.strip():
        raise ValueError("Problem description cannot be empty when extracting constraints.")

    prompt = _build_prompt(description, parameters)
    raw = call_tongyi(prompt, model=model, temperature=temperature)
    constraint_descriptions = _parse_constraints(raw)

    constraints = [
        {"description": desc, "formulation": None, "code": None}
        for desc in constraint_descriptions
    ]

    return ConstraintExtractionResult(constraints=constraints, raw_response=raw, prompt=prompt)


__all__ = [
    "ConstraintExtractionError",
    "ConstraintExtractionResult",
    "get_constraints",
]
