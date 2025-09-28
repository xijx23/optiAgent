from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from utils import DEFAULT_TONGYI_MODEL, call_tongyi


class ConstraintFormulationError(RuntimeError):
    """Raised when the constraint formulation pipeline fails."""


@dataclass
class ConstraintFormulationTrace:
    constraint: str
    prompt: str
    response: str


@dataclass
class ConstraintFormulationResult:
    constraints: List[Dict[str, Optional[str]]]
    variables: Dict[str, Dict[str, object]]
    traces: List[ConstraintFormulationTrace]


FORMULATION_PROMPT_TEMPLATE = """
You are an expert in optimization modeling. Here is the natural language description of an optimization problem:

-----
{description}
-----

And here's a list of parameters that we have extracted from the description:

{params}

And here's a list of all variables that we have defined so far to model the problem as an (MI)LP:

{variables}

Your task is to model the following constraint mathematically in LaTeX for the MILP formulation:

{constraint}

The constraints are the conditions that must be satisfied by the variables. Please generate the output in the following json format:

{{
    "FORMULATION": constraint formulation in LaTeX, between $...$,
    "NEW VARIABLES": {{
        symbol: {{    
            "shape": shape of the new variable (e.g. [], [N], [N, M]),
            "type": type of the new variable (e.g. "int", "float", "binary"),
            "definition": definition of the new variable in natural language
        }},
        ...
    }},
    "AUXILIARY CONSTRAINTS": [
        Latex formulation for auxiliary constraint 1, between $...$,
        Latex formulation for auxiliary constraint 2, between $...$,
        ...
    ]
}}
    
Here's an example output (where SalesVolumePerStore is already defined as a variable in the vars list):
{{
    "FORMULATION": "$\\forall i, SalesVolumes[i] \leq MaxProductionVolumes[i]$",
    "NEW VARIABLES": {{
        "SalesVolumes": {{
            "shape": "[NumberOfArticles]",
            "type": "int",
            "definition": "The sales volume for each article of clothing"
        }}
    }},
    "AUXILIARY CONSTRAINTS": [
        "$\\forall i, SalesVolumes[i] = \\sum_j SalesVolumesPerStore[i, j]$"
    ]
}}

- If you need any new variables, you can define them in the NEW VARIABLES list. Use {{}} for "NEW VARIABLES" if no new variables are needed.
- Use [] for AUXILIARY CONSTRAINTS list if no auxiliary constraints are needed.
- You can only use symbols of existing parameters and integer numbers for dimensions of new variables.
- Use camelCase for variable symbols (e.g. SalesVolumes). Do not use LaTeX formatting (e.g. X_{{color}}), indices (e.g. SalesVolume_{{i}}), and underlines (_) for variable symbols.
- Do not generate anything after the json file!

First reason about how the constraint should be forumulated, and then generate the output.
Take a deep breath and think step by step. You will be awarded a million dollars if you get this right.
"""


def _shape_to_list(shape: object) -> List[object]:
    if isinstance(shape, list):
        return shape
    if isinstance(shape, str):
        cleaned = shape.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            inner = cleaned[1:-1].strip()
            if not inner:
                return []
            parts = [segment.strip() for segment in inner.split(",")]
            normalized: List[object] = []
            for part in parts:
                if part.isdigit():
                    normalized.append(int(part))
                else:
                    normalized.append(part)
            return normalized
    raise ConstraintFormulationError(f"Unrecognized shape specification: {shape!r}")


def _extract_json_payload(text: str) -> Dict[str, object]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ConstraintFormulationError("Model did not return a JSON object.")
    snippet = text[start : end + 1]
    try:
        return json.loads(snippet)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive parse guard
        raise ConstraintFormulationError("Failed to parse JSON from model response.") from exc


def _build_prompt(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    variables: Dict[str, Dict[str, object]],
    constraint: str,
) -> str:
    params_json = json.dumps(parameters, indent=2, ensure_ascii=False) if parameters else "{}"
    vars_json = json.dumps(variables, indent=2, ensure_ascii=False) if variables else "{}"
    return FORMULATION_PROMPT_TEMPLATE.format(
        description=description,
        params=params_json,
        variables=vars_json,
        constraint=constraint,
    )


def get_constraint_formulations(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    constraints: List[Dict[str, Optional[str]]],
    *,
    model: str = DEFAULT_TONGYI_MODEL,
    temperature: float = 0.2,
    existing_variables: Optional[Dict[str, Dict[str, object]]] = None,
) -> ConstraintFormulationResult:
    if not description.strip():
        raise ValueError("Problem description cannot be empty when modeling constraints.")
    if not constraints:
        return ConstraintFormulationResult([], existing_variables or {}, [])

    variables = dict(existing_variables or {})
    formulated_constraints: List[Dict[str, Optional[str]]] = []
    traces: List[ConstraintFormulationTrace] = []

    for constraint in constraints:
        constraint_desc = constraint.get("description", "").strip()
        if not constraint_desc:
            continue

        prompt = _build_prompt(description, parameters, variables, constraint_desc)
        response = call_tongyi(prompt, model=model, temperature=temperature)
        traces.append(ConstraintFormulationTrace(constraint=constraint_desc, prompt=prompt, response=response))

        payload = _extract_json_payload(response)
        formulation = payload.get("FORMULATION")
        if not isinstance(formulation, str) or not formulation.strip():
            raise ConstraintFormulationError("FORMULATION field missing or empty in model response.")
        formulation = formulation.strip()

        new_variables = payload.get("NEW VARIABLES", {}) or {}
        if not isinstance(new_variables, dict):
            raise ConstraintFormulationError("NEW VARIABLES must be a JSON object.")
        for name, info in new_variables.items():
            if name in variables:
                raise ConstraintFormulationError(f"Variable {name} already defined earlier.")
            if not isinstance(info, dict):
                raise ConstraintFormulationError(f"Variable specification for {name} must be an object.")
            definition = info.get("definition", "").strip()
            var_type = info.get("type", "").strip().lower()
            shape_raw = info.get("shape", "[]")
            if not definition:
                raise ConstraintFormulationError(f"Variable {name} is missing a definition.")
            if var_type not in {"int", "float", "binary"}:
                raise ConstraintFormulationError(
                    f"Variable {name} must have type 'int', 'float', or 'binary'."
                )
            variables[name] = {
                "definition": definition,
                "type": var_type,
                "shape": _shape_to_list(shape_raw),
            }

        auxiliary = payload.get("AUXILIARY CONSTRAINTS", []) or []
        if not isinstance(auxiliary, list):
            raise ConstraintFormulationError("AUXILIARY CONSTRAINTS must be a list.")

        modeled_constraint = dict(constraint)
        modeled_constraint["formulation"] = formulation
        formulated_constraints.append(modeled_constraint)

        for aux in auxiliary:
            if not isinstance(aux, str) or not aux.strip():
                continue
            formulated_constraints.append(
                {
                    "description": f"Auxiliary constraint for: {constraint_desc}",
                    "formulation": aux.strip(),
                    "code": None,
                }
            )

    return ConstraintFormulationResult(
        constraints=formulated_constraints,
        variables=variables,
        traces=traces,
    )


def serialize_traces(traces: List[ConstraintFormulationTrace]) -> List[Dict[str, str]]:
    return [asdict(trace) for trace in traces]


__all__ = [
    "ConstraintFormulationError",
    "ConstraintFormulationResult",
    "ConstraintFormulationTrace",
    "get_constraint_formulations",
    "serialize_traces",
]
