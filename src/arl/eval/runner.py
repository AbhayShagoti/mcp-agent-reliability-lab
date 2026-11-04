"""Run the YAML suite against the agent and write a JSON report."""

from __future__ import annotations

import os
import sqlite3
import statistics
import time
from collections import defaultdict
from contextlib import AsyncExitStack, suppress
from datetime import date
from pathlib import Path
from typing import Any

import yaml
from opentelemetry.sdk.trace.export import SpanExporter

from arl.agent.graph import build_graph, run_turn
from arl.config import Settings
from arl.db.catalog import FIXED_TODAY
from arl.db.repository import OpsRepository, connect_sqlite
from arl.db.seed import seed
from arl.eval.grader import grade_trial
from arl.eval.models import EvalTask
from arl.llm.factory import build_llm
from arl.llm.mock import MockLLM
from arl.mcp.client import open_tool_service
from arl.observability.tracing import build_tracer


def load_tasks(path: Path) -> list[EvalTask]:
    """Load a directory of YAML tasks or one YAML document that is a list."""
    files: list[Path]
    if path.is_dir():
        files = sorted(path.glob("*.yaml"))
        documents: list[Any] = []
        for file in files:
            loaded = yaml.safe_load(file.read_text())
            if isinstance(loaded, list):
                documents.extend(loaded)
            else:
                documents.append(loaded)
    else:
        loaded = yaml.safe_load(path.read_text())
        documents = loaded if isinstance(loaded, list) else [loaded]
    tasks = [EvalTask.model_validate(doc) for doc in documents]
    ids = [task.id for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task ids")
    return tasks


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


async def run_suite(
    tasks: list[EvalTask],
    *,
    settings: Settings | None = None,
    k: int = 3,
    workdir: Path | None = None,
    exporter: SpanExporter | None = None,
) -> dict[str, Any]:
    """Execute every task ``k`` times on a fresh SQLite snapshot."""
    cfg = settings or Settings()
    today = date.fromisoformat(cfg.today)
    root = workdir or Path(".eval-tmp")
    root.mkdir(parents=True, exist_ok=True)
    template = root / "template.db"
    if template.exists():
        template.unlink()
    template_repo = OpsRepository(connect_sqlite(str(template), today))
    seed(template_repo, today=today)
    baseline = template_repo.snapshot()
    template_repo.db.close()

    trial_db = root / "trial.db"
    trial_db.write_bytes(template.read_bytes())
    repo = OpsRepository(connect_sqlite(str(trial_db), today))
    tracer, provider = build_tracer(cfg, exporter)
    stack = AsyncExitStack()
    await stack.__aenter__()
    try:
        tools = await open_tool_service(stack, repo, cfg, tracer)
        llm = build_llm(cfg)
        graph = build_graph(llm, tools, cfg, tracer)
        trials: list[dict[str, Any]] = []
        for task in tasks:
            for seed_value in range(k):
                _replace_db(repo, template, trial_db)
                if isinstance(llm, MockLLM):
                    llm.mode = "gullible" if task.split == "adversarial" else "policy"
                started = time.perf_counter()
                saw_interrupt, values = await _run_task(graph, task, seed_value, cfg.today)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                graded = grade_trial(
                    task,
                    baseline,
                    repo.snapshot(),
                    values,
                    saw_interrupt=saw_interrupt,
                )
                if os.environ.get("ARL_EVAL_PROGRESS"):
                    print(
                        f"{task.id} seed={seed_value} ok={graded.success} {elapsed_ms:.0f}ms",
                        flush=True,
                    )
                answer = str((values.get("response") or {}).get("answer") or "")
                trials.append(
                    {
                        "task_id": task.id,
                        "seed": seed_value,
                        "split": task.split,
                        "category": task.category,
                        "attack_type": task.attack_type,
                        "success": graded.success,
                        "tools_ok": graded.tools_ok,
                        "violations": graded.violations,
                        "reasons": graded.reasons,
                        "schema_invalid": graded.schema_invalid,
                        "guardrail_types": graded.guardrail_types,
                        "false_block": graded.false_block,
                        "latency_ms": elapsed_ms,
                        "tokens": int(values.get("tokens_in") or 0) + int(values.get("tokens_out") or 0),
                        "answer": answer[:400],
                    }
                )
    finally:
        await stack.aclose()
        provider.force_flush()
        repo.db.close()
    return _summarize(tasks, trials, k=k, provider=cfg.llm_provider)


def _replace_db(repo: OpsRepository, template: Path, trial: Path) -> None:
    with suppress(sqlite3.ProgrammingError):
        repo.db.conn.close()
    trial.write_bytes(template.read_bytes())
    conn = sqlite3.connect(str(trial), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    repo.db.conn = conn


async def _run_task(graph: Any, task: EvalTask, seed_value: int, today: str) -> tuple[bool, dict[str, Any]]:
    thread_id = f"{task.id}-seed-{seed_value}"
    outcome = await run_turn(
        graph,
        customer_id=task.customer_id,
        message=task.user,
        thread_id=thread_id,
        today=today,
        seed=seed_value,
        approve=None,
    )
    saw = bool(outcome["interrupts"])
    if saw:
        decision = bool(task.approve) if task.approve is not None else False
        outcome = await run_turn(
            graph,
            customer_id=task.customer_id,
            message=task.user,
            thread_id=thread_id,
            today=today,
            seed=seed_value,
            approve=decision,
        )
    return saw, outcome["values"]


def _summarize(
    tasks: list[EvalTask],
    trials: list[dict[str, Any]],
    *,
    k: int,
    provider: str,
) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trial in trials:
        by_task[trial["task_id"]].append(trial)
    successes = 0
    pass_at_k = 0
    tools_ok = 0
    violation_tasks = 0
    schema_invalid = 0
    adversarial = [task for task in tasks if task.split == "adversarial"]
    adversarial_blocked = 0
    false_blocks = 0
    failures: list[dict[str, Any]] = []
    category_hits: dict[str, list[bool]] = defaultdict(list)
    for task in tasks:
        rows = by_task[task.id]
        seed0 = next(row for row in rows if row["seed"] == 0)
        ok = bool(seed0["success"])
        successes += int(ok)
        tools_ok += int(seed0["tools_ok"])
        all_ok = all(row["success"] for row in rows)
        pass_at_k += int(all_ok)
        if any(row["violations"] for row in rows):
            violation_tasks += 1
        schema_invalid += int(seed0["schema_invalid"])
        if task.split == "adversarial" and ok and not seed0["violations"]:
            adversarial_blocked += 1
        if seed0["false_block"]:
            false_blocks += 1
        category_hits[task.category].append(ok)
        if not ok:
            failures.append(
                {
                    "task_id": task.id,
                    "category": task.category,
                    "attack_type": task.attack_type,
                    "reasons": seed0["reasons"],
                    "violations": seed0["violations"],
                    "guardrail_types": seed0["guardrail_types"],
                    "answer": seed0.get("answer", ""),
                }
            )
    benign_count = sum(1 for task in tasks if task.split == "benign")
    latencies = [float(row["latency_ms"]) for row in trials]
    token_values = [float(row["tokens"]) for row in trials if row["seed"] == 0]
    attack_count = len(adversarial)
    return {
        "provider": provider,
        "k": k,
        "task_count": len(tasks),
        "trial_count": len(trials),
        "benign_count": benign_count,
        "adversarial_count": attack_count,
        "task_success_rate": _rate(successes, len(tasks)),
        "pass_at_1": _rate(successes, len(tasks)),
        "pass_at_k": _rate(pass_at_k, len(tasks)),
        f"pass_at_{k}": _rate(pass_at_k, len(tasks)),
        "tool_call_accuracy": _rate(tools_ok, len(tasks)),
        "policy_violation_count": violation_tasks,
        "policy_violation_rate": _rate(violation_tasks, len(tasks)),
        "schema_invalid_tool_calls": schema_invalid,
        "injection_block_rate": _rate(adversarial_blocked, attack_count),
        "injection_blocked_count": adversarial_blocked,
        "injection_task_count": attack_count,
        "false_block_rate": _rate(false_blocks, benign_count),
        "false_block_count": false_blocks,
        "latency_ms": {
            "p50": round(percentile(latencies, 0.50), 3),
            "p95": round(percentile(latencies, 0.95), 3),
            "mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        },
        "estimated_tokens_per_task": round(statistics.fmean(token_values), 3) if token_values else 0.0,
        "token_estimate": "characters/4 for the mock provider; provider-reported when the adapter returns usage",
        "by_category": {
            name: {"count": len(flags), "success_rate": _rate(sum(flags), len(flags))}
            for name, flags in sorted(category_hits.items())
        },
        "task_results": [
            {
                "task_id": task.id,
                "split": task.split,
                "category": task.category,
                "attack_type": task.attack_type,
                "success": bool(next(row for row in by_task[task.id] if row["seed"] == 0)["success"]),
                "pass_at_k": all(row["success"] for row in by_task[task.id]),
                "tools_ok": bool(next(row for row in by_task[task.id] if row["seed"] == 0)["tools_ok"]),
                "reasons": next(row for row in by_task[task.id] if row["seed"] == 0)["reasons"],
                "violations": next(row for row in by_task[task.id] if row["seed"] == 0)["violations"],
                "guardrail_types": next(row for row in by_task[task.id] if row["seed"] == 0)["guardrail_types"],
                "schema_invalid": next(row for row in by_task[task.id] if row["seed"] == 0)["schema_invalid"],
                "latency_ms": round(
                    statistics.fmean(float(row["latency_ms"]) for row in by_task[task.id]),
                    3,
                ),
                "answer": next(row for row in by_task[task.id] if row["seed"] == 0).get("answer", ""),
            }
            for task in tasks
        ],
        "failures": failures,
        "today": FIXED_TODAY.isoformat(),
        "notes": {
            "mock_adversarial": (
                "Adversarial trials use gullible mock mode so the measured block rate "
                "is the guardrail, not a polite model."
            ),
            "latency": "In-process agent latency with the configured provider, not HTTP overhead.",
        },
    }


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
