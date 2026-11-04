"""LangGraph customer-ops agent: guard, plan, verify, act, respond."""

from __future__ import annotations

import json
from datetime import date
from operator import add
from typing import Annotated, Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from opentelemetry.trace import Tracer
from typing_extensions import TypedDict

from arl.agent.intent import classify_intent
from arl.agent.phrases import REFUSAL
from arl.agent.prompts import system_prompt
from arl.config import Settings
from arl.domain.approval import approval_token, token_matches
from arl.domain.models import AgentResponse, Order
from arl.domain.policy import evaluate_refund
from arl.guardrails.allowlist import downgrade_for_untrusted_output, tools_for_intent
from arl.guardrails.injection import detect_injection
from arl.guardrails.output import dump_response, parse_agent_response
from arl.guardrails.pii import scrub_pii
from arl.llm.types import ChatMessage, LLMClient, ToolCall, ToolSchema
from arl.mcp.client import ToolService
from arl.observability.metrics import GUARDRAIL_BLOCKS, LLM_TOKENS

MAX_PLAN_STEPS = 4
SIDE_EFFECTS = frozenset({"create_refund", "send_email", "send_sms"})


class AgentState(TypedDict):
    customer_id: str
    user_text: str
    today: str
    seed: int
    messages: list[dict[str, Any]]
    pending_tool_calls: list[dict[str, Any]]
    tool_trace: Annotated[list[dict[str, Any]], add]
    guardrail_events: Annotated[list[dict[str, Any]], add]
    schema_invalid: Annotated[list[dict[str, Any]], add]
    response: dict[str, Any] | None
    steps: int
    tokens_in: int
    tokens_out: int
    allow_tools: list[str]
    blocked: bool


def initial_state(
    *,
    customer_id: str,
    user_text: str,
    today: str,
    seed: int,
) -> AgentState:
    return {
        "customer_id": customer_id,
        "user_text": user_text,
        "today": today,
        "seed": seed,
        "messages": [],
        "pending_tool_calls": [],
        "tool_trace": [],
        "guardrail_events": [],
        "schema_invalid": [],
        "response": None,
        "steps": 0,
        "tokens_in": 0,
        "tokens_out": 0,
        "allow_tools": [],
        "blocked": False,
    }


def build_graph(
    llm: LLMClient,
    tools: ToolService,
    settings: Settings,
    tracer: Tracer,
) -> Any:
    """Compile the agent. The checkpointer is required for the refund interrupt."""
    threshold = settings.approval_threshold_cents
    secret = settings.approval_secret

    async def input_guard(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("arl.node.input_guard") as span:
            span.set_attribute("arl.customer_id", state["customer_id"])
            scrubbed = scrub_pii(state["user_text"])
            events: list[dict[str, str]] = []
            response: dict[str, Any] | None = None
            if scrubbed.blocked:
                events.append({"type": "pii", "detail": ",".join(scrubbed.kinds)})
                intent = "attack"
                response = dump_response(AgentResponse(answer=REFUSAL, refused=True, refusal_reason="pii"))
            else:
                detection = detect_injection(scrubbed.text)
                if detection.matched:
                    events.append({"type": "injection", "detail": detection.rule or "injection"})
                    intent = "attack"
                    response = dump_response(
                        AgentResponse(
                            answer=REFUSAL,
                            refused=True,
                            refusal_reason=detection.rule,
                        )
                    )
                else:
                    intent = classify_intent(scrubbed.text)
            for event in events:
                GUARDRAIL_BLOCKS.labels(rule=event["type"]).inc()
            messages = [
                {
                    "role": "system",
                    "content": system_prompt(
                        customer_id=state["customer_id"],
                        today=state["today"],
                        threshold_cents=threshold,
                    ),
                },
                {"role": "user", "content": scrubbed.text},
            ]
            span.set_attribute("arl.intent", intent)
            return {
                "user_text": scrubbed.text,
                "messages": messages,
                "allow_tools": sorted(tools_for_intent(intent)),
                "guardrail_events": events,
                "blocked": response is not None,
                "response": response,
            }

    async def plan(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("arl.node.plan") as span:
            if state.get("response"):
                return {}
            allowed = set(state["allow_tools"])
            schemas = [
                ToolSchema(
                    name=spec.name,
                    description=spec.description,
                    parameters=spec.parameters,
                )
                for spec in tools.specs.values()
                if spec.name in allowed
            ]
            result = await llm.complete(_chat(state["messages"]), schemas, state["seed"])
            LLM_TOKENS.labels(direction="input").inc(result.input_tokens)
            LLM_TOKENS.labels(direction="output").inc(result.output_tokens)
            span.set_attribute("arl.tool_call_count", len(result.tool_calls))
            messages = list(state["messages"])
            steps = state["steps"] + 1
            tokens_in = state["tokens_in"] + result.input_tokens
            tokens_out = state["tokens_out"] + result.output_tokens
            if result.tool_calls:
                pending = [
                    {"id": call.id, "name": call.name, "arguments": call.arguments} for call in result.tool_calls
                ]
                messages.append({"role": "assistant", "content": result.content, "tool_calls": pending})
                return {
                    "messages": messages,
                    "pending_tool_calls": pending,
                    "steps": steps,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                }
            parsed, _repaired = parse_agent_response(result.content or "{}")
            messages.append({"role": "assistant", "content": result.content})
            return {
                "messages": messages,
                "pending_tool_calls": [],
                "response": dump_response(parsed),
                "steps": steps,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
            }

    async def verify(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("arl.node.verify"):
            events: list[dict[str, str]] = []
            invalid: list[dict[str, str]] = []
            accepted: list[dict[str, Any]] = []
            messages = list(state["messages"])
            allowed = set(state["allow_tools"])
            for call in state["pending_tool_calls"]:
                name = str(call["name"])
                if name not in allowed:
                    events.append({"type": "allowlist", "detail": name})
                    messages.append(_tool_message(call, {"ok": False, "error": "tool_not_allowed"}))
                    continue
                arguments = dict(call["arguments"])
                schema = tools.specs[name].parameters
                if "customer_id" in schema.get("properties", {}):
                    arguments["customer_id"] = state["customer_id"]
                error = _schema_error(schema, arguments)
                if error:
                    invalid.append({"tool": name, "error": error})
                    messages.append(_tool_message(call, {"ok": False, "error": "schema_invalid"}))
                    continue
                accepted.append({**call, "arguments": arguments})

            needs_approval = [
                call
                for call in accepted
                if call["name"] == "create_refund" and int(call["arguments"]["amount_cents"]) > threshold
            ]
            if needs_approval:
                decision = interrupt(
                    {
                        "type": "refund_approval",
                        "order_id": needs_approval[0]["arguments"]["order_id"],
                        "amount_cents": needs_approval[0]["arguments"]["amount_cents"],
                        "reason": needs_approval[0]["arguments"]["reason"],
                    }
                )
                approved = isinstance(decision, dict) and bool(decision.get("approved"))
                if not approved:
                    events.append({"type": "approval_denied", "detail": "create_refund"})
                    denied_ids = {call["id"] for call in needs_approval}
                    for call in needs_approval:
                        messages.append(_tool_message(call, {"ok": False, "error": "approval_denied"}))
                    accepted = [call for call in accepted if call["id"] not in denied_ids]
                else:
                    stamped: list[dict[str, Any]] = []
                    approved_ids = {call["id"] for call in needs_approval}
                    for call in accepted:
                        if call["id"] not in approved_ids:
                            stamped.append(call)
                            continue
                        arguments = dict(call["arguments"])
                        arguments["approval_token"] = approval_token(
                            secret,
                            str(arguments["order_id"]),
                            str(arguments["customer_id"]),
                            int(arguments["amount_cents"]),
                        )
                        stamped.append({**call, "arguments": arguments})
                    accepted = stamped
            for event in events:
                GUARDRAIL_BLOCKS.labels(rule=event["type"]).inc()
            if invalid:
                GUARDRAIL_BLOCKS.labels(rule="schema_invalid").inc(len(invalid))
            return {
                "messages": messages,
                "pending_tool_calls": accepted,
                "guardrail_events": events,
                "schema_invalid": invalid,
            }

    async def act(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("arl.node.act"):
            messages = list(state["messages"])
            trace: list[dict[str, Any]] = []
            events: list[dict[str, str]] = []
            allow = set(state["allow_tools"])
            injected = False
            for call in state["pending_tool_calls"]:
                name = str(call["name"])
                arguments = dict(call["arguments"])
                if injected and name in SIDE_EFFECTS:
                    events.append({"type": "allowlist", "detail": name})
                    messages.append(_tool_message(call, {"ok": False, "error": "tool_not_allowed"}))
                    trace.append(_trace(call, status="blocked", detail="untrusted_tool_output"))
                    continue
                if name == "create_refund":
                    blocked = await _refund_policy_block(tools, state, arguments, threshold, secret)
                    if blocked:
                        events.append({"type": "policy", "detail": blocked})
                        messages.append(_tool_message(call, {"ok": False, "error": blocked}))
                        trace.append(_trace(call, status="blocked", detail=blocked))
                        continue
                payload = await tools.call(name, arguments)
                trace.append(_trace(call, status="ok" if payload.get("ok") is not False else "error"))
                if detect_injection(json.dumps(payload)).matched:
                    injected = True
                    events.append({"type": "tool_output_injection", "detail": name})
                    allow = downgrade_for_untrusted_output(allow)
                messages.append(_tool_message(call, payload))
            for event in events:
                GUARDRAIL_BLOCKS.labels(rule=event["type"]).inc()
            return {
                "messages": messages,
                "pending_tool_calls": [],
                "tool_trace": trace,
                "guardrail_events": events,
                "allow_tools": sorted(allow),
            }

    def respond(state: AgentState) -> dict[str, Any]:
        with tracer.start_as_current_span("arl.node.respond"):
            current = state.get("response")
            if current is None:
                current = dump_response(AgentResponse(answer=REFUSAL, refused=True, refusal_reason="incomplete"))
            parsed, _repaired = parse_agent_response(json.dumps(current))
            return {"response": dump_response(parsed)}

    graph: StateGraph[AgentState] = StateGraph(AgentState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("plan", plan)
    graph.add_node("verify", verify)
    graph.add_node("act", act)
    graph.add_node("respond", respond)
    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        lambda state: "respond" if state.get("response") else "plan",
        {"respond": "respond", "plan": "plan"},
    )
    graph.add_conditional_edges("plan", _route_plan, {"respond": "respond", "verify": "verify"})
    graph.add_conditional_edges(
        "verify",
        _route_verify,
        {"act": "act", "plan": "plan", "respond": "respond"},
    )
    graph.add_conditional_edges("act", _route_act, {"plan": "plan", "respond": "respond"})
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=InMemorySaver())


def _route_plan(state: AgentState) -> str:
    if state.get("response"):
        return "respond"
    if state["steps"] >= MAX_PLAN_STEPS:
        return "respond"
    if state.get("pending_tool_calls"):
        return "verify"
    return "respond"


def _route_verify(state: AgentState) -> str:
    if state.get("pending_tool_calls"):
        return "act"
    if state["steps"] >= MAX_PLAN_STEPS:
        return "respond"
    return "plan"


def _route_act(state: AgentState) -> str:
    if state["steps"] >= MAX_PLAN_STEPS:
        return "respond"
    return "plan"


def _chat(messages: list[dict[str, Any]]) -> list[ChatMessage]:
    converted: list[ChatMessage] = []
    for message in messages:
        calls = [
            ToolCall(id=str(call["id"]), name=str(call["name"]), arguments=dict(call["arguments"]))
            for call in message.get("tool_calls", [])
        ]
        converted.append(
            ChatMessage(
                role=str(message["role"]),
                content=str(message.get("content") or ""),
                tool_calls=calls,
                tool_call_id=message.get("tool_call_id"),
                name=message.get("name"),
            )
        )
    return converted


def _tool_message(call: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": call["name"],
        "tool_call_id": call["id"],
        "content": json.dumps(payload),
    }


def _trace(call: dict[str, Any], *, status: str, detail: str = "") -> dict[str, Any]:
    return {
        "id": call["id"],
        "name": call["name"],
        "arguments": call["arguments"],
        "status": status,
        "source": "model",
        "detail": detail,
    }


def _schema_error(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    try:
        Draft202012Validator(schema).validate(arguments)
    except ValidationError as exc:
        return str(exc.message)
    return None


async def _refund_policy_block(
    tools: ToolService,
    state: AgentState,
    arguments: dict[str, Any],
    threshold: int,
    secret: str,
) -> str | None:
    order_id = str(arguments["order_id"])
    customer_id = str(arguments["customer_id"])
    amount = int(arguments["amount_cents"])
    payload = await tools.call(
        "get_order",
        {"order_id": order_id, "customer_id": customer_id},
    )
    order = _order_from_tool(payload)
    has_approval = amount <= threshold or token_matches(
        secret,
        order_id,
        customer_id,
        amount,
        arguments.get("approval_token") if isinstance(arguments.get("approval_token"), str) else None,
    )
    decision = evaluate_refund(
        order,
        amount,
        date.fromisoformat(state["today"]),
        has_existing_refund=bool(order and order.refunds),
        threshold_cents=threshold,
        has_approval=has_approval,
    )
    return decision.reason


def _order_from_tool(payload: dict[str, Any]) -> Order | None:
    if not payload.get("found"):
        return None
    raw = payload.get("order")
    if not isinstance(raw, dict):
        return None
    return Order.model_validate(raw)


async def run_turn(
    graph: Any,
    *,
    customer_id: str,
    message: str,
    thread_id: str,
    today: str,
    seed: int = 0,
    approve: bool | None = None,
) -> dict[str, Any]:
    """Run one user message, resuming a pending refund approval when ``approve`` is set."""
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 40}
    if approve is not None:
        result = await graph.ainvoke(Command(resume={"approved": approve}), config)
    else:
        result = await graph.ainvoke(
            initial_state(customer_id=customer_id, user_text=message, today=today, seed=seed),
            config,
        )
    interrupts = list(result.get("__interrupt__", []))
    values = {key: value for key, value in result.items() if key != "__interrupt__"}
    return {"values": values, "interrupts": interrupts}
