VENV ?= .venv
SYSTEM_PYTHON ?= python3
PYTHON ?= $(VENV)/bin/python
PIP ?= $(PYTHON) -m pip
PYTHONPATH ?= src
VENV_STAMP := $(VENV)/.hierreco-venv-installed

.PHONY: venv install install-dev reinstall reinstall-dev lint format check stats clean-venv

venv: $(VENV_STAMP)

$(VENV_STAMP):
	@if [ -e "$(VENV)" ]; then \
		echo "Existing $(VENV) was not created by the current venv setup."; \
		echo "Run: make clean-venv"; \
		exit 1; \
	fi
	$(SYSTEM_PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	touch $(VENV_STAMP)

install: venv
	$(PIP) install -e ".[visualization]"

install-dev: venv
	$(PIP) install -e ".[visualization,dev]"

reinstall: clean-venv
	$(MAKE) install

reinstall-dev: clean-venv
	$(MAKE) install-dev

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff format .

check:
	$(PYTHON) -m compileall -q -x '(^|/)(\._|cache/|runs/|__pycache__/)' src
	$(PYTHON) -m ruff check .

stats:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m hierreco_nn.cli.compute_dataset_stats --bad-limit 0

clean-venv:
	rm -rf $(VENV)
