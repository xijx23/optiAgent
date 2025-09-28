from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Files that mirror OptiMUS naming so later stages can reuse them.
DESCRIPTION_FILENAME = "desc.txt"
INITIAL_STATE_FILENAME = "state_0_description.json"


@dataclass
class ProblemPaths:
    """Convenience container for all files associated with a problem."""

    root: Path

    @property
    def description_file(self) -> Path:
        return self.root / DESCRIPTION_FILENAME

    @property
    def initial_state_file(self) -> Path:
        return self.root / INITIAL_STATE_FILENAME


def _prompt_for_description() -> str:
    """Interactively collect a multi-line problem description from stdin."""

    print(
        "Enter the optimization problem description. Type EOF on a new line when done:",
        flush=True,
    )
    lines: list[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "EOF":
            break
        lines.append(line)
    description = "\n".join(lines).strip()
    if not description:
        raise ValueError("Problem description cannot be empty.")
    return description


def collect_problem_description(description: Optional[str] = None) -> str:
    """Return a sanitized problem description from provided text or stdin."""

    if description is not None:
        cleaned = description.strip()
        if not cleaned:
            raise ValueError("Problem description cannot be empty.")
        return cleaned
    return _prompt_for_description()


def store_problem_description(
    problem_name: str,
    *,
    base_dir: Path | str = "problems",
    description: Optional[str] = None,
) -> dict:
    """Persist the raw description and return the initial OptiMUS-style state."""

    problem_root = Path(base_dir) / problem_name
    problem_root.mkdir(parents=True, exist_ok=True)
    paths = ProblemPaths(problem_root)

    desc_text = collect_problem_description(description)
    
    params_path = problem_root / "params.json"
    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    state = {
        "description": desc_text,
        "parameters": params,
        "meta": {
            "problem_name": problem_name,
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    }
    paths.initial_state_file.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "state": state,
        "paths": {
            "description": str(paths.description_file),
            "initial_state": str(paths.initial_state_file),
        },
    }


__all__ = [
    "collect_problem_description",
    "store_problem_description",
    "ProblemPaths",
]
