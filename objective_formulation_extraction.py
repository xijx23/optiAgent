from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

from utils import DEFAULT_TONGYI_MODEL, call_tongyi


class ObjectiveFormulationError(RuntimeError):
    """Raised when the objective formulation pipeline fails."""


@dataclass
class ObjectiveFormulationResult:
    objective: Dict[str, Optional[str]]
    prompt: str
    response: str


OBJECTIVE_FORM_PROMPT = """
You are an expert in optimization modeling. Convert the following objective into a LaTeX optimization expression.

Problem description:
-----
{description}
-----

Known parameters (JSON):
{params}

Known decision variables defined so far (JSON):
{variables}

Objective (natural language):
"{objective}"

Return ONLY a JSON object:
{{
  "FORMULATION": "<LaTeX expression wrapped in $$ ... $$>",
  "CODE": null
}}

Guidelines:
- Use existing parameters/variables only.
- Keep the LaTeX within $$ ... $$ and ensure it represents a maximize/minimize statement.
- Do not include explanatory text outside the JSON object.
"""


def _build_prompt(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    variables: Dict[str, Dict[str, object]],
    objective_description: str,
) -> str:
    params_json = json.dumps(parameters, indent=2, ensure_ascii=False) if parameters else "{}"
    vars_json = json.dumps(variables, indent=2, ensure_ascii=False) if variables else "{}"
    return OBJECTIVE_FORM_PROMPT.format(
        description=description,
        params=params_json,
        variables=vars_json,
        objective=objective_description,
    )


def _extract_formulation(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ObjectiveFormulationError("Model did not return a JSON object.")
    payload = json.loads(text[start : end + 1])
    formulation = payload.get("FORMULATION")
    if not isinstance(formulation, str) or not formulation.strip():
        raise ObjectiveFormulationError("FORMULATION field missing or empty.")
    return formulation.strip()


def get_objective_formulation(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    variables: Dict[str, Dict[str, object]],
    objective: Dict[str, Optional[str]],
    *,
    model: str = DEFAULT_TONGYI_MODEL,
    temperature: float = 0.2,
) -> ObjectiveFormulationResult:
    if not description.strip():
        raise ValueError("Problem description cannot be empty when modeling objective.")
    objective_description = (objective or {}).get("description", "").strip()
    if not objective_description:
        raise ObjectiveFormulationError("Objective description missing.")

    prompt = _build_prompt(description, parameters, variables, objective_description)
    response = call_tongyi(prompt, model=model, temperature=temperature)
    formulation = _extract_formulation(response)

    return ObjectiveFormulationResult(
        objective={
            "description": objective_description,
            "formulation": formulation,
            "code": None,
        },
        prompt=prompt,
        response=response,
    )


__all__ = [
    "ObjectiveFormulationError",
    "ObjectiveFormulationResult",
    "get_objective_formulation",
]
