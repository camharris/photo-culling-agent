.PHONY: help lint lint-check format install-dev pre-commit test clean

help:
	@echo "Available commands:"
	@echo "  make install-dev      Install development dependencies"
	@echo "  make format           Format code with isort and black"
	@echo "  make lint             Run all linters (autoflake, isort, black, flake8)"
	@echo "  make lint-check       Check linting without modifying files"
	@echo "  make pre-commit       Install pre-commit hooks"
	@echo "  make test             Run tests"
	@echo "  make clean            Clean up cache files"

install-dev:
	pip install -r requirements-dev.txt

format:
	isort src tests scripts
	black src tests scripts

lint:
	python scripts/lint.py

lint-check:
	python scripts/lint.py --check

pre-commit:
	pre-commit install

test:
	pytest tests

clean:
	rm -rf .pytest_cache
	rm -rf src/__pycache__
	rm -rf src/**/__pycache__
	rm -rf tests/__pycache__
	rm -rf tests/**/__pycache__
