from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List



def _normalize_dimension_expression(dim: object) -> str:
    if isinstance(dim, int):
        return f"range({dim})"
    if isinstance(dim, float):
        return f"range(int({dim}))"
    if isinstance(dim, str):
        return f"_index_iter({dim})"
    raise ValueError(f"Unsupported dimension descriptor: {dim!r}")


def _gurobi_vtype(var_type: str) -> str:
    mapping = {
        "float": "CONTINUOUS",
        "continuous": "CONTINUOUS",
        "int": "INTEGER",
        "integer": "INTEGER",
        "binary": "BINARY",
    }
    return mapping.get((var_type or "").lower(), "CONTINUOUS")


@dataclass
class AssemblyResult:
    script_path: Path
    data_path: Path
    script: str


def assemble_solver_script(
    state: Dict[str, object],
    problem_dir: Path | str,
    *,
    script_name: str = "solve_model.py",
) -> AssemblyResult:
    root = Path(problem_dir)
    root.mkdir(parents=True, exist_ok=True)

    parameters: Dict[str, Dict[str, object]] = state.get("parameters", {}) or {}
    variables: Dict[str, Dict[str, object]] = state.get("variables", {}) or {}
    constraints: List[Dict[str, object]] = state.get("constraints", []) or []
    objective: Dict[str, object] = state.get("objective", {}) or {}

    data_payload = {
        name: spec.get("value")
        for name, spec in parameters.items()
    }
    data_path = root / "data.json"
    data_path.write_text(
        json.dumps(data_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines: List[str] = []
    lines.append("#!/usr/bin/env python3")
    lines.append("from __future__ import annotations")
    lines.append("")
    lines.append("import json")
    lines.append("from pathlib import Path")
    lines.append("")
    lines.append("from gurobipy import Model, GRB, quicksum")
    lines.append("")
    lines.append("")
    lines.append("def _index_iter(obj):")
    lines.append("    if isinstance(obj, int):")
    lines.append("        return range(obj)")
    lines.append("    if isinstance(obj, float):")
    lines.append("        return range(int(obj))")
    lines.append("    if isinstance(obj, dict):")
    lines.append("        return list(obj.keys())")
    lines.append("    if hasattr(obj, '__len__'):")
    lines.append("        return range(len(obj))")
    lines.append("    raise TypeError(f'Unsupported dimension source: {obj!r}')")
    lines.append("")
    lines.append("")
    lines.append("def main() -> None:")
    lines.append("    data_path = Path(__file__).with_name('data.json')")
    lines.append("    data = json.loads(data_path.read_text(encoding='utf-8'))")
    lines.append("")
    if parameters:
        lines.append("    # Parameters")
        for name in parameters:
            lines.append(f"    {name} = data.get('{name}')")
        lines.append("")

    lines.append("    model = Model('OptiAgentModel')")
    lines.append("    model.Params.OutputFlag = 1")
    lines.append("")

    if variables:
        lines.append("    # Decision variables")
        for name, spec in variables.items():
            var_type = _gurobi_vtype(spec.get("type", "float"))
            shape = spec.get("shape", []) or []
            if not shape:
                lines.append(
                    f"    {name} = model.addVar(vtype=GRB.{var_type}, name='{name}')"
                )
            else:
                dim_exprs = [
                    _normalize_dimension_expression(dim)
                    for dim in shape
                ]
                lines.append(
                    f"    {name} = model.addVars({', '.join(dim_exprs)}, vtype=GRB.{var_type}, name='{name}')"
                )
        lines.append("")

    lines.append("    # Constraints")
    for constraint in constraints:
        code = constraint.get("code")
        if not code:
            continue
        for raw in str(code).splitlines():
            lines.append("    " + raw)
    lines.append("")

    lines.append("    # Objective")
    objective_code = objective.get("code")
    if objective_code:
        for raw in str(objective_code).splitlines():
            lines.append("    " + raw)
    else:
        lines.append("    # Objective code missing")
    lines.append("")
    lines.append("    model.optimize()")
    lines.append("    status = model.Status")
    lines.append("    if status == GRB.OPTIMAL:")
    lines.append("        print('Optimal objective:', model.objVal)")
    lines.append("        solution = {var.VarName: var.X for var in model.getVars()}")
    lines.append("        print('Solution:', json.dumps(solution, indent=2))")
    lines.append("        output_path = Path(__file__).with_name('output_solution.json')")
    lines.append("        output_path.write_text(json.dumps({'objective': model.objVal, 'solution': solution}, indent=2), encoding='utf-8')")
    lines.append("    else:")
    lines.append("        print(f'Model finished with status {status}')")
    lines.append("")
    lines.append("")
    lines.append("if __name__ == '__main__':")
    lines.append("    main()")
    lines.append("")

    script = '\n'.join(lines)

    script_path = root / script_name
    script_path.write_text(script, encoding='utf-8')

    return AssemblyResult(script_path=script_path, data_path=data_path, script=script)


__all__ = [
    'AssemblyResult',
    'assemble_solver_script',
]
