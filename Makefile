UV ?= uv

.PHONY: install lint format typecheck test eval gate seed up down loadtest render ci

install:
	$(UV) sync --all-groups --frozen

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck:
	$(UV) run mypy --strict src

test:
	$(UV) run pytest --cov=arl --cov-report=term-missing --cov-report=json:reports/coverage.json --cov-fail-under=80

eval:
	$(UV) run python -m arl.eval.cli --output reports/eval_report.json --k 3
	$(UV) run python -m arl.eval.gate --report reports/eval_report.json

seed:
	$(UV) run python data/seed.py

up:
	docker compose up --build -d

down:
	docker compose down -v

loadtest:
	$(UV) run python scripts/loadtest.py

render:
	$(UV) run python scripts/render_results.py

ci: lint typecheck test eval
