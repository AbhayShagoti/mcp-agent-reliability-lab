"""Turn a local-model eval report into a failure analysis.

Every count in the markdown comes from the JSON report. The script does not
invent rates.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Reasons that are wording or the refused flag. Everything else is a tool,
# state, interrupt, or schema miss.
WORDING = ("missing:", "refused_flag")


def _load(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(path)
    return payload


def _is_wording(reason: str) -> bool:
    return reason.startswith("missing:") or reason == "refused_flag"


def _has_refund_miss(reasons: list[str]) -> bool:
    return any(reason.startswith("refund") or reason == "unexpected_refund" for reason in reasons)


def _state_ok(reasons: list[str]) -> bool:
    return all(_is_wording(reason) for reason in reasons)


def build_analysis(report: dict[str, object]) -> str:
    results = report.get("task_results")
    if not isinstance(results, list):
        raise ValueError("report has no task_results; re-run with the current harness")
    rows = [row for row in results if isinstance(row, dict)]
    latency = report.get("latency_ms")
    if not isinstance(latency, dict):
        latency = {}
    lines = [
        f"Provider `{report.get('provider')}` on {report.get('task_count')} tasks, k={report.get('k')}, "
        f"{report.get('trial_count')} trials.",
        "",
        "Strict task success uses the same grader as CI, including the canned phrases the mock emits. "
        "A trial can move the database correctly and still fail that check.",
        "",
        "## By category",
        "",
        "| Category | Tasks | Strict success | Tools matched | State ok (wording ignored) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    by_cat: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_cat.setdefault(str(row["category"]), []).append(row)
    state_ok_n = 0
    for name in sorted(by_cat):
        group = by_cat[name]
        strict = sum(1 for row in group if row["success"])
        tools = sum(1 for row in group if row["tools_ok"])
        state = sum(1 for row in group if _state_ok(list(row["reasons"])))  # type: ignore[arg-type]
        state_ok_n += state
        lines.append(f"| {name} | {len(group)} | {strict} | {tools} | {state} |")
    lines.append("")
    lines.append(
        f"State-ok across the suite: {state_ok_n}/{len(rows)}. "
        "State-ok means the only misses, if any, are `response_contains` phrases or the refused flag."
    )
    refund_ok = [row for row in rows if row["category"] == "refund_ok"]
    refund_blocked = [row for row in rows if row["category"] in {"refund_denied", "refund_denied_human"}]
    issued = sum(1 for row in refund_ok if not _has_refund_miss(list(row["reasons"])))  # type: ignore[arg-type]
    held = sum(1 for row in refund_blocked if not _has_refund_miss(list(row["reasons"])))  # type: ignore[arg-type]
    lines.append(
        f"Of {len(refund_ok)} refunds the task wanted written, {issued} have no `refund_count` "
        "error, so the expected refund row was present. "
        f"Of {len(refund_blocked)} refunds the task forbade, {held} also have no refund-row error, "
        "so the model did not create one. Those still fail the strict grader when the tool list "
        "or the exact refusal sentence does not match."
    )
    lines.append("")
    violated = [row for row in rows if row["violations"]]
    if violated:
        lines.append("Tasks whose seed-0 result carries a policy-violation tag:")
        lines.append("")
        for row in violated:
            lines.append(f"- `{row['task_id']}`: {row['violations']}. Reasons: {row['reasons']}.")
        lines.append("")
    lines.append("## Why strict failures happen")
    lines.append("")
    reason_counts: Counter[str] = Counter()
    for row in rows:
        if row["success"]:
            continue
        seen: set[str] = set()
        for reason in row["reasons"]:  # type: ignore[union-attr]
            text = str(reason)
            key = "missing phrase" if text.startswith("missing:") else text
            if key not in seen:
                reason_counts[key] += 1
                seen.add(key)
    lines.append("| Reason (seed 0) | Failed tasks |")
    lines.append("| --- | ---: |")
    for key, count in reason_counts.most_common():
        lines.append(f"| {key} | {count} |")
    lines.append("")
    lines.append("## Adversarial tasks and which guardrail fired")
    lines.append("")
    lines.append(
        "Input-filter blocks return in a few milliseconds because the model is not called. "
        "Trials at or above 500 ms reached the model. That cutoff is from this run: "
        "regex blocks were about 2 ms and model turns were several seconds."
    )
    lines.append("")
    lines.append("| Attack | Tasks | Strict success | Reached model | Guardrail events | Violations |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: |")
    attacks: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        if row["split"] != "adversarial":
            continue
        attacks.setdefault(str(row["attack_type"]), []).append(row)
    for name in sorted(attacks):
        group = attacks[name]
        reached = sum(1 for row in group if float(row["latency_ms"]) >= 500)  # type: ignore[arg-type]
        guards: Counter[str] = Counter()
        violations = 0
        for row in group:
            for event in row["guardrail_types"]:  # type: ignore[union-attr]
                guards[str(event)] += 1
            if row["violations"]:
                violations += 1
        guard_text = ", ".join(f"{event} {count}" for event, count in guards.most_common()) or "none"
        strict = sum(1 for row in group if row["success"])
        lines.append(f"| {name} | {len(group)} | {strict} | {reached} | {guard_text} | {violations} |")
    lines.append("")
    lines.append("## Examples of strict failures")
    lines.append("")
    shown = 0
    for row in rows:
        if row["success"] or shown >= 8:
            continue
        answer = str(row.get("answer") or "").replace("\n", " ")
        lines.append(f"- `{row['task_id']}` ({row['category']}): {row['reasons']}. Answer: {answer[:220]}")
        shown += 1
    if shown == 0:
        lines.append("- None.")
    lines.append("")
    lines.append(
        f"End-to-end latency over {report.get('trial_count')} trials: "
        f"p50 {latency.get('p50')} ms, p95 {latency.get('p95')} ms, mean {latency.get('mean')} ms. "
        f"Provider-reported tokens per task (seed 0 mean): {report.get('estimated_tokens_per_task')}."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="reports/qwen25_1_5b.json")
    parser.add_argument("--output", default="reports/qwen25_1_5b_analysis.md")
    args = parser.parse_args()
    text = build_analysis(_load(ROOT / args.report if not Path(args.report).is_absolute() else Path(args.report)))
    out = ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
