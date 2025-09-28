#!/usr/bin/env python3
"""CLI pipeline for natural-language optimization modeling with OptiAgent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from constraint_extraction import get_constraints
from constraint_formulation_extraction import get_constraint_formulations, serialize_traces as serialize_constraint_traces
from objective_formulation_extraction import get_objective_formulation
from code_generation import get_codes, serialize_traces as serialize_code_traces
from code_assembly import assemble_solver_script
from code_execution import execute_generated_code
from objective_extraction import get_objective
from parameter_extraction import extract_and_store_parameters
from problem_input import store_problem_description
from utils import load_json, save_json

STATE_0_FILE = "state_0_description.json"
STATE_1_FILE = "state_1_params.json"
STATE_2_FILE = "state_2_objective.json"
STATE_3_FILE = "state_3_constraints.json"
PARAMS_JSON = "params.json"
OBJECTIVE_LOG = "objective_extraction_log.json"
CONSTRAINTS_LOG = "constraints_extraction_log.json"
STATE_4_FILE = "state_4_constraints_modeled.json"
CONSTRAINT_FORM_LOG = "constraint_formulations_log.json"
STATE_5_FILE = "state_5_objective_modeled.json"
OBJECTIVE_FORM_LOG = "objective_formulation_log.json"
STATE_6_FILE = "state_6_code.json"
CODE_LOG = "code_generation_log.json"
GENERATED_SCRIPT = "solve_model.py"
EXECUTION_LOG = "code_execution_output.txt"
DATA_FILE = "data.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect problem description, extract parameters, and derive optimization objective.",
    )
    parser.add_argument(
        "-n",
        "--name",
        dest="problem_name",
        required=True,
        help="Problem name (used as sub-directory under base dir).",
    )
    parser.add_argument(
        "-b",
        "--base-dir",
        dest="base_dir",
        default="problems",
        help="Root directory for storing problem artifacts (default: problems).",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-d",
        "--desc",
        dest="desc",
        help="Inline natural-language description (multi-line via escaped newlines).",
    )
    group.add_argument(
        "-f",
        "--desc-file",
        dest="desc_file",
        help="Path to UTF-8 text file containing the problem description.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing state files without confirmation.",
    )
    return parser.parse_args()


def read_description_from_file(path: str | Path) -> str:
    file_path = Path(path)
    if not file_path.exists():
        print(f"[error] 描述文件不存在: {file_path}", file=sys.stderr)
        sys.exit(2)
    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as exc:  # pragma: no cover - defensive IO
        print(f"[error] 读取描述文件失败: {exc}", file=sys.stderr)
        sys.exit(2)


def confirm_overwrite_if_needed(problem_root: Path, force: bool) -> None:
    existing_states = [STATE_0_FILE, STATE_1_FILE, STATE_2_FILE, PARAMS_JSON]
    existing_states.extend([STATE_3_FILE, CONSTRAINTS_LOG])
    existing_states.extend([STATE_4_FILE, CONSTRAINT_FORM_LOG])
    existing_states.extend([STATE_5_FILE, OBJECTIVE_FORM_LOG])
    existing_states.extend([STATE_6_FILE, CODE_LOG])
    existing_states.extend([GENERATED_SCRIPT, EXECUTION_LOG, DATA_FILE])
    if any((problem_root / name).exists() for name in existing_states) and not force:
        answer = input(
            f"[warn] 目标目录已存在历史状态: {problem_root}\n覆盖写入? [y/N]: "
        ).strip().lower()
        if answer != "y":
            print("已取消。")
            sys.exit(1)


def main() -> int:
    args = parse_args()

    base_dir = Path(args.base_dir)
    problem_name = args.problem_name
    problem_root = base_dir / problem_name

    if args.desc_file:
        description_override = read_description_from_file(args.desc_file)
    else:
        description_override = args.desc

    confirm_overwrite_if_needed(problem_root, force=args.force)

    # Step 1: Persist raw description (state_0)
    try:
        state_payload = store_problem_description(
            problem_name=problem_name,
            base_dir=base_dir,
            description=description_override,
        )
    except Exception as exc:
        print(f"[error] 写入问题描述失败: {exc}", file=sys.stderr)
        return 3

    state0_path = Path(state_payload["paths"]["initial_state"])
    state0 = load_json(state0_path)

    # Step 2: Extract parameters and store state_1
    params_result = extract_and_store_parameters(
        Path(""),
        description=state0["description"],
        model="qwen-plus",
    )
    state0["parameters"] = params_result.parameters
    state1_path = problem_root / STATE_1_FILE
    save_json(state0, state1_path)
    print("[info] 参数提取完成，结果已写入", state1_path)

    # Step 3: Derive objective using Tongyi and store state_2
    state = load_json(state1_path)
    objective_result = get_objective(
        state["description"],
        state.get("parameters", {}),
        model="qwen-plus",
    )
    print("[objective]", objective_result.objective)
    state["objective"] = objective_result.objective
    state2_path = problem_root / STATE_2_FILE
    save_json(state, state2_path)

    (problem_root / OBJECTIVE_LOG).write_text(
        json.dumps(
            {
                "prompt": objective_result.prompt,
                "response": objective_result.raw_response,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Step 4: Extract constraints and store state_3
    state = load_json(state2_path)
    constraints_result = get_constraints(
        state["description"],
        state.get("parameters", {}),
        model="qwen-plus",
    )
    print("[constraints]", constraints_result.constraints)
    state["constraints"] = constraints_result.constraints
    state3_path = problem_root / STATE_3_FILE
    save_json(state, state3_path)

    (problem_root / CONSTRAINTS_LOG).write_text(
        json.dumps(
            {
                "prompt": constraints_result.prompt,
                "response": constraints_result.raw_response,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Step 5: Convert constraints to LaTeX formulations and store state_4
    state = load_json(state3_path)
    constraint_models = get_constraint_formulations(
        state["description"],
        state.get("parameters", {}),
        state.get("constraints", []),
        model="qwen-plus",
    )
    print("[constraints-formulation]", constraint_models.constraints)
    state["constraints"] = constraint_models.constraints
    state["variables"] = constraint_models.variables
    state4_path = problem_root / STATE_4_FILE
    save_json(state, state4_path)

    (problem_root / CONSTRAINT_FORM_LOG).write_text(
        json.dumps(
            serialize_constraint_traces(constraint_models.traces),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Step 6: Convert objective to LaTeX and store state_5
    state = load_json(state4_path)
    objective_modeled = get_objective_formulation(
        state["description"],
        state.get("parameters", {}),
        state.get("variables", {}),
        state.get("objective", {}),
        model="qwen-plus",
    )
    print("[objective-formulation]", objective_modeled.objective)
    state["objective"] = objective_modeled.objective
    state5_path = problem_root / STATE_5_FILE
    save_json(state, state5_path)

    (problem_root / OBJECTIVE_FORM_LOG).write_text(
        json.dumps(
            {
                "prompt": objective_modeled.prompt,
                "response": objective_modeled.response,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


    # Step 7: Generate solver code for constraints and objective
    state = load_json(state5_path)
    code_generation = get_codes(
        state["description"],
        state.get("parameters", {}),
        state.get("variables", {}),
        state.get("constraints", []),
        state.get("objective", {}),
        model="qwen-plus",
    )
    print("[code-constraints]", code_generation.constraints)
    print("[code-objective]", code_generation.objective)
    state["constraints"] = code_generation.constraints
    state["objective"] = code_generation.objective
    state6_path = problem_root / STATE_6_FILE
    save_json(state, state6_path)

    (problem_root / CODE_LOG).write_text(
        json.dumps(
            serialize_code_traces(code_generation.traces),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Step 8: Assemble runnable solver script
    state = load_json(state6_path)
    assembly = assemble_solver_script(
        state,
        problem_root,
        script_name=GENERATED_SCRIPT,
    )
    print("[code-assembly]", assembly.script_path)

    # Step 9: Execute generated script
    execution_result = execute_generated_code(assembly.script_path)
    run_log_path = problem_root / EXECUTION_LOG
    run_log_path.write_text(
        json.dumps(
            {
                "returncode": execution_result.returncode,
                "stdout": execution_result.stdout,
                "stderr": execution_result.stderr,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    status_tag = "success" if execution_result.succeeded else f"failure (returncode={execution_result.returncode})"
    print(f"[code-execution] {status_tag}")

    print()
    print("[ok] 处理完成。生成的关键文件如下：")
    print(f"  - 描述文件::            {state_payload['paths']['description']}")
    print(f"  - 状态״̬ {STATE_0_FILE}: {state0_path}")
    print(f"  - 状态״̬ {STATE_1_FILE}: {state1_path}")
    print(f"  - 状态״̬ {STATE_2_FILE}: {state2_path}")
    print(f"  - 状态״̬ {STATE_3_FILE}: {state3_path}")
    print(f"  - 状态״̬ {STATE_4_FILE}: {state4_path}")
    print(f"  - 状态״̬ {STATE_5_FILE}: {state5_path}")
    print(f"  - 状态״̬ {STATE_6_FILE}: {state6_path}")
    print(f"  - 参数信息文件:        {assembly.data_path}")
    print(f"  - 参数信息文件:        {assembly.script_path}")
    print(f"  - 参数提取日志:        {problem_root / 'params_extraction_log.json'}")
    print(f"  - 目标提取日志:        {problem_root / OBJECTIVE_LOG}")
    print(f"  - 目标建模日志:        {problem_root / OBJECTIVE_FORM_LOG}")
    print(f"  - 约束提取日志:        {problem_root / CONSTRAINTS_LOG}")
    print(f"  - 约束建模日志:        {problem_root / CONSTRAINT_FORM_LOG}")
    print(f"  - 代码生成日志:        {problem_root / CODE_LOG}")
    print(f"  - 代码运行日志:        {run_log_path}")



    return 0


if __name__ == "__main__":
    sys.exit(main())
