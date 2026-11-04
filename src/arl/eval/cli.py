"""Command line for the eval suite and the merge gate."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from arl.config import Settings
from arl.eval.gate import evaluate_gate, load_json
from arl.eval.runner import load_tasks, run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent Reliability Lab eval")
    parser.add_argument("--tasks", default="evals/tasks")
    parser.add_argument("--output", default="reports/eval_report.json")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--ids", default="", help="comma-separated task ids")
    parser.add_argument("--ids-file", default="", help="file with one task id per line")
    parser.add_argument("--gate", action="store_true", help="apply the gate after the run")
    parser.add_argument("--thresholds", default="evals/thresholds.json")
    parser.add_argument("--baseline", default="evals/baseline.json")
    parser.add_argument("--report", default="", help="grade an existing report instead of running")
    args = parser.parse_args()
    if args.report:
        report = load_json(Path(args.report))
    else:
        tasks = load_tasks(Path(args.tasks))
        wanted: set[str] = set()
        if args.ids:
            wanted.update(item.strip() for item in args.ids.split(",") if item.strip())
        if args.ids_file:
            wanted.update(line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip())
        if wanted:
            tasks = [task for task in tasks if task.id in wanted]
        report = asyncio.run(run_suite(tasks, settings=Settings(), k=args.k))
        report["generated_at"] = datetime.now(UTC).isoformat()
        report["versions"] = _versions()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n")
        _write_markdown(report, output.with_suffix(".md"))
        print(
            f"tasks={report['task_count']} success={report['task_success_rate']} "
            f"tools={report['tool_call_accuracy']} "
            f"injection_block={report['injection_block_rate']} "
            f"violations={report['policy_violation_count']} "
            f"pass_at_{args.k}={report['pass_at_k']}"
        )
    if args.gate or args.report:
        errors = evaluate_gate(
            report,
            load_json(Path(args.thresholds)),
            load_json(Path(args.baseline)),
        )
        if errors:
            for error in errors:
                print(f"GATE FAIL: {error}")
            raise SystemExit(1)
        print("GATE PASS")


def gate_main() -> None:
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


def _versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in ("langgraph", "mcp", "fastapi", "pydantic"):
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = "unknown"
    return found


def _write_markdown(report: dict[str, object], path: Path) -> None:
    lines = [
        "# Eval report",
        "",
        f"- Provider: `{report['provider']}`",
        f"- Tasks: {report['task_count']} (k={report['k']}, trials={report['trial_count']})",
        f"- Task success: {report['task_success_rate']}",
        f"- pass^k: {report['pass_at_k']}",
        f"- Tool-call accuracy: {report['tool_call_accuracy']}",
        f"- Policy violations: {report['policy_violation_count']}",
        f"- Schema-invalid tool calls: {report['schema_invalid_tool_calls']}",
        f"- Injection block rate: {report['injection_block_rate']}",
        f"- False-block rate: {report['false_block_rate']}",
        f"- In-process latency p50/p95 ms: {report['latency_ms']}",
        f"- Estimated tokens per task: {report['estimated_tokens_per_task']}",
        "",
        "Failures:",
        "",
    ]
    failures = report["failures"]
    if not isinstance(failures, list) or not failures:
        lines.append("- none")
    else:
        for item in failures:
            if isinstance(item, dict):
                lines.append(f"- {item.get('task_id')}: {item.get('reasons')}")
    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
