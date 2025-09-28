from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from utils import DEFAULT_TONGYI_MODEL, call_tongyi


class CodeGenerationError(RuntimeError):
    """Raised when code generation fails."""


@dataclass
class CodeGenerationTrace:
    kind: str
    target: str
    prompt: str
    response: str


@dataclass
class CodeGenerationResult:
    constraints: List[Dict[str, Optional[str]]]
    objective: Dict[str, Optional[str]]
    traces: List[CodeGenerationTrace]


CONSTRAINT_PROMPT_TEMPLATE = """
You are an expert in mathematical programming. Convert the following constraint into solver-ready {solver} code in Python.

Problem description:
-----
{description}
-----

Relevant parameters (JSON):
{params}

Current decision variables (JSON):
{variables}

Constraint to implement (JSON):
{constraint}

Implementation requirements:
- Assume the Gurobi model has already been instantiated as `model`.
- Only emit the code that creates this constraint. Do not include imports, parameter loading, or variable creation.
- Use `model.addConstr` / `model.addConstrs` as appropriate.
- For multi-dimensional variables, use Gurobi's tuple indexing (e.g., `Var[i, j]`).
- For parameter arrays, use Python list indexing (e.g., `Param[i][j]`).

Return your answer strictly in this format:
CODE
=====
<python code implementing the constraint>
=====

Do not add explanations before or after the block.
"""


OBJECTIVE_PROMPT_TEMPLATE = """
You are an expert in mathematical programming. Convert the following optimization objective into solver-ready {solver} code in Python.

Problem description:
-----
{description}
-----

Relevant parameters (JSON):
{params}

Current decision variables (JSON):
{variables}

Objective to implement (JSON):
{objective}

Implementation requirements:
- Assume the Gurobi model has already been instantiated as `model`.
- Only emit the code that defines the objective (no imports or variable declarations).
- Use `model.setObjective` with `GRB.MAXIMIZE` or `GRB.MINIMIZE` as appropriate.
- Employ `quicksum` for summations when helpful.

Return your answer strictly in this format:
CODE
=====
<python code implementing the objective>
=====

Do not add explanations before or after the block.
"""


def _extract_code_block(text: str) -> str:
    start = text.find('=====')
    end = text.find('=====', start + 5)
    if start != -1 and end != -1:
        snippet = text[start + 5 : end].strip()
    else:
        start = text.find('```')
        end = text.find('```', start + 3)
        if start == -1 or end == -1:
            raise CodeGenerationError('Failed to locate code block in model response.')
        snippet = text[start + 3 : end].strip()
    snippet = snippet.replace('```python', '').replace('```', '').strip()
    while snippet.startswith('====='):
        snippet = snippet[5:].strip()
    while snippet.endswith('====='):
        snippet = snippet[:-5].strip()
    return snippet





def _build_constraint_prompt(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    variables: Dict[str, Dict[str, object]],
    constraint: Dict[str, Optional[str]],
    solver: str,
) -> str:
    return CONSTRAINT_PROMPT_TEMPLATE.format(
        solver=solver,
        description=description,
        params=json.dumps(parameters, indent=2, ensure_ascii=False) if parameters else "{}",
        variables=json.dumps(variables, indent=2, ensure_ascii=False) if variables else "{}",
        constraint=json.dumps(constraint, indent=2, ensure_ascii=False),
    )


def _build_objective_prompt(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    variables: Dict[str, Dict[str, object]],
    objective: Dict[str, Optional[str]],
    solver: str,
) -> str:
    return OBJECTIVE_PROMPT_TEMPLATE.format(
        solver=solver,
        description=description,
        params=json.dumps(parameters, indent=2, ensure_ascii=False) if parameters else "{}",
        variables=json.dumps(variables, indent=2, ensure_ascii=False) if variables else "{}",
        objective=json.dumps(objective, indent=2, ensure_ascii=False),
    )


def get_codes(
    description: str,
    parameters: Dict[str, Dict[str, object]],
    variables: Dict[str, Dict[str, object]],
    constraints: List[Dict[str, Optional[str]]],
    objective: Dict[str, Optional[str]],
    *,
    solver: str = "gurobipy",
    model: str = DEFAULT_TONGYI_MODEL,
    temperature: float = 0.2,
) -> CodeGenerationResult:
    if not description.strip():
        raise ValueError("Problem description cannot be empty when generating code.")

    traces: List[CodeGenerationTrace] = []
    coded_constraints: List[Dict[str, Optional[str]]] = []

    for constraint in constraints:
        constraint_desc = constraint.get("description", "").strip()
        if not constraint_desc:
            continue
        prompt = _build_constraint_prompt(description, parameters, variables, constraint, solver)
        response = call_tongyi(prompt, model=model, temperature=temperature)
        code = _extract_code_block(response)

        modeled = dict(constraint)
        modeled["code"] = code
        coded_constraints.append(modeled)
        traces.append(CodeGenerationTrace(kind="constraint", target=constraint_desc, prompt=prompt, response=response))

    objective_desc = (objective or {}).get("description", "").strip()
    if not objective_desc:
        raise CodeGenerationError("Objective description missing for code generation.")
    prompt = _build_objective_prompt(description, parameters, variables, objective, solver)
    response = call_tongyi(prompt, model=model, temperature=temperature)
    code = _extract_code_block(response)

    coded_objective = dict(objective)
    coded_objective["code"] = code

    traces.append(CodeGenerationTrace(kind="objective", target=objective_desc, prompt=prompt, response=response))

    return CodeGenerationResult(constraints=coded_constraints, objective=coded_objective, traces=traces)


def serialize_traces(traces: List[CodeGenerationTrace]) -> List[Dict[str, str]]:
    return [asdict(trace) for trace in traces]


__all__ = [
    "CodeGenerationError",
    "CodeGenerationTrace",
    "CodeGenerationResult",
    "get_codes",
    "serialize_traces",
]
