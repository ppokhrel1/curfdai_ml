# Testing Guide

## Running Tests

```bash
# Install dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html
open htmlcov/index.html

# Run specific test
pytest tests/unit/test_llm_service.py::test_llm_generate_json -v

# Run with detailed output
pytest -vv --tb=long