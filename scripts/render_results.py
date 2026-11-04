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
REAL_START = "<!-- REAL:START -->"
REAL_END = "<!-- REAL:END -->"
ANALYSIS_START = "<!-- ANALYSIS:START -->"
ANALYSIS_END = "<!-- ANALYSIS:END -->"


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
        "This table is the CI baseline. The mock is deterministic, so pass^3 equals pass^1. "
        "It checks the harness, not a neural model. The Qwen table below is the local-model run."
    )
    return "\n".join(lines)


def build_real_table(report: dict[str, object]) -> str:
    latency = report["latency_ms"]
    if not isinstance(latency, dict):
        raise ValueError("latency section missing")
    artifact = "reports/qwen25_1_5b.json"
    rows = [
        ("Task success (pass^1)", _pct(report["task_success_rate"])),
        ("pass^3", _pct(report["pass_at_3"])),
        ("Tool-call accuracy", _pct(report["tool_call_accuracy"])),
        ("Policy violations", str(report["policy_violation_count"])),
        ("Schema-invalid tool calls", str(report["schema_invalid_tool_calls"])),
        ("Injection block rate", _pct(report["injection_block_rate"])),
        ("False-block rate", _pct(report["false_block_rate"])),
        ("End-to-end p50 / p95", f"{latency['p50']} ms / {latency['p95']} ms"),
        ("Provider tokens / task", str(report["estimated_tokens_per_task"])),
    ]
    lines = [
        "Qwen2.5-1.5B-Instruct, GGUF Q4_K_M, llama.cpp b12099, 4 CPU threads, no GPU. "
        f"{report['task_count']} tasks, k={report['k']}, {report['trial_count']} trials. "
        "Same grader as the mock table.",
        "",
        "| Metric | Value | Artifact |",
        "| --- | --- | --- |",
    ]
    for name, value in rows:
        lines.append(f"| {name} | {value} | `{artifact}` |")
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
    real_path = ROOT / "reports" / "qwen25_1_5b.json"
    if real_path.exists() and REAL_START in updated and REAL_END in updated:
        real = build_real_table(_load(real_path))
        head, tail = updated.split(REAL_START, 1)
        _, tail = tail.split(REAL_END, 1)
        updated = f"{head}{REAL_START}\n{real}\n{REAL_END}{tail}"
    analysis_path = ROOT / "reports" / "qwen25_1_5b_analysis.md"
    if analysis_path.exists() and ANALYSIS_START in updated and ANALYSIS_END in updated:
        analysis = analysis_path.read_text().strip()
        head, tail = updated.split(ANALYSIS_START, 1)
        _, tail = tail.split(ANALYSIS_END, 1)
        updated = f"{head}{ANALYSIS_START}\n{analysis}\n{ANALYSIS_END}{tail}"
    readme.write_text(updated)
    print(table)


if __name__ == "__main__":
    main()
