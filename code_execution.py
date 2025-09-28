from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def execute_generated_code(
    script_path: Path | str,
    *,
    timeout: Optional[int] = 120,
) -> ExecutionResult:
    path = Path(script_path)
    if not path.exists():
        raise FileNotFoundError(path)

    process = subprocess.run(
        ["python", path.name],
        cwd=path.parent,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return ExecutionResult(
        returncode=process.returncode,
        stdout=process.stdout,
        stderr=process.stderr,
    )


__all__ = [
    "ExecutionResult",
    "execute_generated_code",
]
