# Agent Reliability Lab

A customer-operations agent for a small shop: order status, shipments, refunds, and policy questions. The agent plans with a LangGraph loop, and every tool call goes through one of three MCP servers. FastAPI serves it over server-sent events. A 150-task eval, including 25 adversarial tasks, gates pull requests.

The default model is a deterministic mock, so `make test` and CI need no API key. Ollama and a Hugging Face transformer's path are optional. OpenAI and Anthropic adapters exist and stay off unless `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set.

The shop data is synthetic. `data/seed.py` uses Faker for display names and a fixed catalog for ids, money, and dates, so the eval does not move when a Faker release changes a name. The reference clock is 1 November 2026.

## Architecture

```mermaid
flowchart LR
  U[Client / Gradio] -->|HTTP + SSE| API[FastAPI gateway]
  API --> G[LangGraph agent]
  G --> GR[Guardrails]
  G --> LLM[LLM adapter]
  G -->|MCP| M1[orders server]
  G -->|MCP| M2[policy server]
  G -->|MCP| M3[notify server]
  API --> OT[OpenTelemetry] --> J[Jaeger]
  API --> P[/metrics/] --> PR[Prometheus] --> GF[Grafana]
```

The graph is `input_guard -> plan -> verify -> act -> plan`, then `respond`. `verify` checks the tool allow-list and the JSON schema, and it interrupts for a human decision when a refund is over $50. `act` calls MCP. After a tool result that contains an injection payload, side-effect tools are removed from the allow-list for the rest of the turn.

The orders server enforces the refund rules again, and it ignores any customer id the model sends: the gateway overwrites `customer_id` with the authenticated customer. Email and SMS are delivered only to the address or phone on that customer's record. A refund over $50 also needs an HMAC the model cannot mint.

Policy search is hybrid. BM25 (weight 0.85) runs in process over eight policy documents. The other 0.15 is cosine similarity of a 64-dimensional hashing-trick embedding. On Postgres the same vectors are stored in pgvector and the cosine leg is the `<#>`/`<=>` distance. This is not a neural encoder. It is a free, deterministic stand-in so CI does not download embedding weights.

## Quickstart

Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --all-groups --frozen
make seed          # sqlite data/ops.db
make test          # ruff is separate; this is pytest at coverage >= 80
make eval          # 150 tasks, k=3, then the gate
make up            # API, three MCP servers, Postgres, Redis, Jaeger, Prometheus, Grafana
```

Chat against the local API (mock model, token `dev-token`):

```bash
curl -N http://127.0.0.1:8000/v1/chat \
  -H 'Authorization: Bearer dev-token' \
  -H 'Content-Type: application/json' \
  -d '{"message":"What is the status of order ORD-1001?","customer_id":"CUST-1001"}'
```

Grafana is on port 3000, Jaeger on 16686, Prometheus on 9090. A refund above $50 returns an `approval_required` event; resume with the same `thread_id` and `"approve": true`.

Useful environment variables:

| Variable | Default | Role |
| --- | --- | --- |
| `ARL_LLM_PROVIDER` | `mock` | `mock`, `ollama`, `openai`, `anthropic` |
| `ARL_DATABASE_URL` | `sqlite:///./data/ops.db` | SQLite, or Postgres in Compose |
| `ARL_OLLAMA_MODEL` | `qwen2.5:1.5b-instruct` | Used when the provider is `ollama` |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` | unset | Optional. The process refuses to start those providers without a key |
| `ARL_OTEL_ENDPOINT` | unset | OTLP gRPC, for example `http://jaeger:4317` |

## Eval methodology

Each task is a YAML file in `evals/tasks/` with the user text, the authenticated customer, the tools that must be called, the expected database delta, and any guardrail event that must fire. `scripts/generate_tasks.py` rebuilds the 150 files from the catalog. The committed files are what CI runs.

The grader diffs a fresh SQLite snapshot against the database after the turn. A new refund, a changed order status, or a notification that was not expected is a task failure. A refund that was not expected, a notification to any address other than the customer's own, or a body that contains `attacker@` is also a **policy violation**. The gate fails on any policy violation and on any schema-invalid tool call.

**pass^3** reruns every task with seeds 0, 1, and 2. The mock ignores the seed, so pass^3 matches pass^1. The loop is still executed so a later stochastic provider can be scored the same way.

Adversarial tasks use a second mock mode, `gullible`, which obeys injection text in the user message or in a tool result. Benign tasks use `policy` mode, which follows the refund rules. The injection-block rate is therefore the share of the 25 adversarial tasks whose attack did not land and whose guardrail event fired. It is a closed-set measurement of this regex detector plus the allow-list, the refund verifier, and the notification binding. It is not an estimate of attacks in the wild.

The optional tone judge in `arl.eval.judge` is not part of the gate and was not used for the table below.

Thresholds live in `evals/thresholds.json`. The baseline rates live in `evals/baseline.json`. CI fails if task success or the injection-block rate drops more than 3 points below that baseline, if either falls under its floor, if any policy violation appears, or if a schema-invalid tool call appears.

`reports/eval_report.json` and `reports/eval_report.md` are the k=3 mock run from this environment. `reports/loadtest.json` is 200 requests at concurrency 10 against the mock API, with the rate limit raised so the sample is not HTTP 429s. `reports/coverage.json` is pytest-cov on `src/arl`. Regenerate the README table with `make render` after those three files change.

## Results

<!-- RESULTS:START -->
| Metric | Value | Artifact |
| --- | --- | --- |
| Task success (pass^1) | 100.0% | `reports/eval_report.json` |
| pass^3 | 100.0% | `reports/eval_report.json` |
| Tool-call accuracy | 100.0% | `reports/eval_report.json` |
| Policy violations | 0 | `reports/eval_report.json` |
| Schema-invalid tool calls | 0 | `reports/eval_report.json` |
| Injection block rate | 100.0% | `reports/eval_report.json` |
| False-block rate | 0.0% | `reports/eval_report.json` |
| In-process p50 / p95 (mock) | 5.332 ms / 11.827 ms | `reports/eval_report.json` |
| Estimated tokens / task (mock, chars/4) | 622.693 | `reports/eval_report.json` |
| API overhead p50 / p95 (mock) | 56.973 ms / 80.059 ms | `reports/loadtest.json` |
| Requests / second (mock API) | 167.383 | `reports/loadtest.json` |
| Line+branch coverage | 85% | `reports/coverage.json` |

Local-model latency and task success were not measured in the environment that produced this table. The nightly workflow runs that subset when Ollama is available.
<!-- RESULTS:END -->

In-process latency is the agent loop on SQLite with the mock. API latency is the same agent behind uvicorn. Neither number is a local neural model. The nightly workflow (`.github/workflows/nightly.yml`) runs 40 tasks from `evals/nightly40.txt` on Ollama `qwen2.5:1.5b-instruct` and uploads `reports/nightly/`. That job has not produced a file in this environment, so there is no local-model success rate or latency to report.

## Design decisions

- **Mock for the merge gate.** A small local model is too noisy to be a regression test for tool schemas and state diffs. The mock is a deterministic planner. Regressions in routing, schemas, or guardrails fail the gate. Model quality is left to the nightly Ollama job.
- **Gullible mock on adversarial tasks.** A polite mock would "pass" injection tests by never attempting the attack. The gullible mode attempts the refund or the email. The guardrail has to stop it.
- **Rules live in one function.** `evaluate_refund` is used by the verifier and by the orders MCP server, so the two cannot drift.
- **Authz is not a model argument.** `customer_id` on a tool call is replaced with the authenticated customer before the call. Notification tools do not accept a destination.
- **Hash embeddings.** pgvector is wired up for the policy server, and the vectors are real. They are not MiniLM or OpenAI embeddings, because those would add a download or a paid call to CI.
- **SQLite for the gate, Postgres in Compose.** Snapshotting SQLite is a file copy, which keeps 450 trials (150 × 3) fast. Compose uses Postgres so the pgvector leg is actually exercised in the stack.

## Limitations

- The injection detector is a regex list. A paraphrase that avoids those phrases can pass it. The refund rules and the notification binding still apply, but the published block rate only covers the 25 committed attacks.
- Hashing-trick vectors do not capture paraphrases the way a trained encoder would. BM25 carries the policy-QA tasks.
- Mock-mode success is harness correctness. It does not say how Qwen2.5-1.5B would score. That number is not in this README because it was not measured here.
- The API checkpointer is in-memory. A restart drops approval threads.
- The Space demo does not resume an approval interrupt.
- Terraform is validated in CI and is not applied. Applying it creates a Fargate service, RDS, and a $5 budget alarm, and it costs money. Destroy it afterwards.
- Redis rate limiting is per token in one process, or in Redis when `ARL_REDIS_URL` is set. It is not an edge limiter.

## How to deploy

**Laptop.** `make up`.

**Kind.** Build `arl:local`, then `bash deploy/k8s/smoke.sh`. CI does this on every pull request. The kind overlay runs the API with SQLite and the mock model.

**Hugging Face Space (free, ZeroGPU).** Create a Gradio Space and set the repository secrets `HF_TOKEN` and `HF_SPACE_ID`. `.github/workflows/deploy-space.yml` uploads `app/space/` on a release. The Space requirements install this package from GitHub, so the repository has to be public first. The UI defaults to the mock. The Qwen2.5-1.5B option calls `transformers` under `spaces.GPU` when the Space runtime is present.

**Render free tier.** `deploy/render.yaml` runs the container in mock mode. A 512 MB instance cannot load a 1.5B model.

**AWS (optional, costs money).** From `deploy/terraform/aws`:

```bash
terraform init
terraform apply -var image_tag=... -var db_password=...
```

The stack is a small VPC, ECR, ECS Fargate (256 CPU / 512 MB), RDS Postgres `db.t4g.micro`, a private S3 bucket for reports, and a $5 monthly budget alarm. CI runs `terraform validate` and `tflint` only.

## Resume bullets

These use only the mock-mode numbers in the table above.

- Built a customer-operations agent on LangGraph with 3 MCP servers (orders, hybrid BM25/pgvector policy search, notifications), Pydantic structured outputs, and a human-approval interrupt for refunds over $50, served by FastAPI with SSE; packaged with Docker Compose, Kustomize, and Terraform for ECS Fargate.
- Designed a 150-task state-diff eval with pass^3 and 25 adversarial tasks and gated GitHub Actions on it, measuring 100% task success, 100% tool-call accuracy, 0 policy violations, a 100% injection block rate, and a 0% false-block rate on the committed mock suite.
- Instrumented each agent node and tool call with OpenTelemetry and Prometheus/Grafana. On the mock path, in-process p50/p95 was 5.332 ms / 11.827 ms and HTTP API p50/p95 was 56.973 ms / 80.059 ms at 167.383 requests/second, with 85% line and branch coverage.
