from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from utils import DEFAULT_TONGYI_MODEL, call_tongyi


class ObjectiveExtractionError(RuntimeError):
    """Raised when the objective extraction pipeline fails."""


@dataclass
class ObjectiveExtractionResult:
    objective: Dict[str, Optional[str]]
    raw_response: str
    prompt: str


OBJECTIVE_PROMPT_TEMPLATE = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

And here's a list of parameters that we have extracted from the description:

{params}

Your task is to identify and extract the optimization objective from the description. The objective is the goal that the optimization model is trying to achieve (e.g. maximize profit, minimize cost). Please generate the output in the following format:

=====
OBJECTIVE: objective description
=====

for example:

=====    
OBJECTIVE: "The goal is to maximize the total profit from producing television sets"
=====

- Do not generate anything after the objective.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


def _build_prompt(description: str, parameters: Dict[str, Dict[str, object]]) -> str:
    params_json = json.dumps(parameters, indent=4, ensure_ascii=False) if parameters else "{}"
    return OBJECTIVE_PROMPT_TEMPLATE.format(description=description, params=params_json)


def _extract_objective_text(text: str) -> str:
    start = text.find("=====")
    if start == -1:
        raise ObjectiveExtractionError("No opening delimiter found in model response.")
    end = text.find("=====", start + 5)
    if end == -1:
        raise ObjectiveExtractionError("No closing delimiter found in model response.")
    core = text[start + 5 : end].strip()
    if core.upper().startswith("OBJECTIVE:"):
        core = core[len("OBJECTIVE:") :].strip()
    if not core:
        raise ObjectiveExtractionError("Objective text is empty after parsing.")
    return core


def get_objective(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    *,
    model: str = DEFAULT_TONGYI_MODEL,
    temperature: float = 0.2,
) -> ObjectiveExtractionResult:
    if not description.strip():
        raise ValueError("Problem description cannot be empty when extracting objective.")

    prompt = _build_prompt(description, parameters)
    raw = call_tongyi(prompt, model=model, temperature=temperature)
    objective_text = _extract_objective_text(raw)

    objective = {
        "description": objective_text,
        "formulation": None,
        "code": None,
    }

    return ObjectiveExtractionResult(objective=objective, raw_response=raw, prompt=prompt)


__all__ = [
    "ObjectiveExtractionError",
    "ObjectiveExtractionResult",
    "get_objective",
]

