.PHONY: setup check smoke grid
setup:
	uv sync --locked
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run pytest -q
smoke:
	uv run landcover smoke --output outputs/synthetic-smoke
grid:
	uv run landcover grid --output outputs/grid
