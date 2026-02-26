## Setup
Dev setup (linting & hooks)
```bash
pip install -e ".[dev]"
pre-commit install  # sets up git hooks
# on demand: pre-commit run --all-files
```

## Tests

- Run the main suite: `python -m pytest tests`
- Run API-level tests only: `python -m pytest tests/api`



