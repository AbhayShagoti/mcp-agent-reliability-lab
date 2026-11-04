"""Compare an eval report to the thresholds and baseline stored in the repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def evaluate_gate(
    report: dict[str, Any],
    thresholds: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    """Return human-readable failures. An empty list means the gate passed."""
    errors: list[str] = []
    success = float(report["task_success_rate"])
    base_success = float(baseline["task_success_rate"])
    drop = float(thresholds["max_success_drop_from_baseline"])
    if success < base_success - drop:
        errors.append(f"task success {success:.4f} dropped more than {drop:.4f} below baseline {base_success:.4f}")
    minimum = float(thresholds["min_task_success_rate"])
    if success < minimum:
        errors.append(f"task success {success:.4f} is below the minimum {minimum:.4f}")

    blocked = float(report["injection_block_rate"])
    base_blocked = float(baseline["injection_block_rate"])
    block_drop = float(thresholds["max_injection_block_drop_from_baseline"])
    if blocked < base_blocked - block_drop:
        errors.append(
            f"injection block rate {blocked:.4f} dropped more than {block_drop:.4f} below baseline {base_blocked:.4f}"
        )
    min_block = float(thresholds["min_injection_block_rate"])
    if blocked < min_block:
        errors.append(f"injection block rate {blocked:.4f} is below the minimum {min_block:.4f}")

    violations = int(report["policy_violation_count"])
    max_violations = int(thresholds["max_policy_violations"])
    if violations > max_violations:
        errors.append(f"policy violations {violations} exceed {max_violations}")

    invalid = int(report["schema_invalid_tool_calls"])
    max_invalid = int(thresholds["max_schema_invalid_tool_calls"])
    if invalid > max_invalid:
        errors.append(f"schema-invalid tool calls {invalid} exceed {max_invalid}")

    false_block = float(report["false_block_rate"])
    max_false = float(thresholds["max_false_block_rate"])
    if false_block > max_false:
        errors.append(f"false-block rate {false_block:.4f} exceeds {max_false:.4f}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply the eval gate to a report")
    parser.add_argument("--report", default="reports/eval_report.json")
    parser.add_argument("--thresholds", default="evals/thresholds.json")
    parser.add_argument("--baseline", default="evals/baseline.json")
    args = parser.parse_args()
    errors = evaluate_gate(
        load_json(Path(args.report)),
        load_json(Path(args.thresholds)),
        load_json(Path(args.baseline)),
    )
    if errors:
        for error in errors:
            print(f"GATE FAIL: {error}")
        raise SystemExit(1)
    print("GATE PASS")


if __name__ == "__main__":
    main()
