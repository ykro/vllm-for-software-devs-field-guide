PYTHON ?= $(shell if [ -x ../../.venv-shared/bin/python ]; then echo ../../.venv-shared/bin/python; else command -v python3; fi)
RUFF ?= $(shell if [ -x ../../.venv-shared/bin/ruff ]; then echo ../../.venv-shared/bin/ruff; else command -v ruff; fi)
OUT ?= 2000000
IN ?= 20000000
HOURS ?= 8

.PHONY: all verify calc lint

all: lint verify

verify:
	$(PYTHON) scripts/verify_field_guide.py

calc:
	$(PYTHON) scripts/self_host_or_api.py --input-tokens-per-day $(IN) --output-tokens-per-day $(OUT) --busy-hours $(HOURS)

lint:
	$(RUFF) check .
