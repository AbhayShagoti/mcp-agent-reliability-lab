"""Fill the README results table from committed report files.

The table is the only place numbers are allowed to appear, and every cell is
read from reports/ so it cannot drift from the artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def _pct(value: object) -> str:
    return f"{float(value) * 100:.1f}%"


def build_table(eval_report: dict[str, object], load: dict[str, object], coverage: dict[str, object]) -> str:
    latency = eval_report["latency_ms"]
    http = load["latency_ms"]
    if not isinstance(latency, dict) or not isinstance(http, dict):
        raise ValueError("latency sections are missing")
    totals = coverage.get("totals", {})
    if not isinstance(totals, dict):
        raise ValueError("coverage totals missing")
    covered = totals.get("percent_covered_display") or totals.get("percent_covered")
    rows = [
        ("Task success (pass^1)", _pct(eval_report["task_success_rate"]), "reports/eval_report.json"),
        ("pass^3", _pct(eval_report["pass_at_3"]), "reports/eval_report.json"),
        ("Tool-call accuracy", _pct(eval_report["tool_call_accuracy"]), "reports/eval_report.json"),
        ("Policy violations", str(eval_report["policy_violation_count"]), "reports/eval_report.json"),
        ("Schema-invalid tool calls", str(eval_report["schema_invalid_tool_calls"]), "reports/eval_report.json"),
        ("Injection block rate", _pct(eval_report["injection_block_rate"]), "reports/eval_report.json"),
        ("False-block rate", _pct(eval_report["false_block_rate"]), "reports/eval_report.json"),
        (
            "In-process p50 / p95 (mock)",
            f"{latency['p50']} ms / {latency['p95']} ms",
            "reports/eval_report.json",
        ),
        (
            "Estimated tokens / task (mock, chars/4)",
            str(eval_report["estimated_tokens_per_task"]),
            "reports/eval_report.json",
        ),
        (
            "API overhead p50 / p95 (mock)",
            f"{http['p50']} ms / {http['p95']} ms",
            "reports/loadtest.json",
        ),
        ("Requests / second (mock API)", str(load["rps"]), "reports/loadtest.json"),
        ("Line+branch coverage", f"{covered}%", "reports/coverage.json"),
    ]
    lines = [
        "| Metric | Value | Artifact |",
        "| --- | --- | --- |",
    ]
    for name, value, source in rows:
        lines.append(f"| {name} | {value} | `{source}` |")
    lines.append("")
    lines.append(
        "Local-model latency and task success were not measured in the environment "
        "that produced this table. The nightly workflow runs that subset when Ollama is available."
    )
    return "\n".join(lines)


def main() -> None:
    table = build_table(
        _load(ROOT / "reports" / "eval_report.json"),
        _load(ROOT / "reports" / "loadtest.json"),
        _load(ROOT / "reports" / "coverage.json"),
    )
    readme = ROOT / "README.md"
    text = readme.read_text()
    if START not in text or END not in text:
        raise SystemExit("README is missing RESULTS markers")
    before, rest = text.split(START, 1)
    _, after = rest.split(END, 1)
    updated = f"{before}{START}\n{table}\n{END}{after}"
    readme.write_text(updated)
    print(table)


if __name__ == "__main__":
    main()
