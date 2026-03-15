PYTHON ?= python3
CONFIG_PHASE1 ?= configs/phase1.yaml
CONFIG_PHASE2 ?= configs/phase2.yaml
CONFIG_PHASE3 ?= configs/phase3.yaml
CONFIG_SMOKE ?= configs/smoke.yaml

.PHONY: setup test smoke phase1 phase2 phase3 report clean

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.lock
	$(PYTHON) -m pip install -e . --no-deps

test:
	$(PYTHON) -m pytest

smoke:
	$(PYTHON) -m src.cli smoke --config $(CONFIG_SMOKE)

phase1:
	$(PYTHON) -m src.cli phase1 --config $(CONFIG_PHASE1)

phase2:
	$(PYTHON) -m src.cli phase2 --config $(CONFIG_PHASE2)

phase3:
	$(PYTHON) -m src.cli phase3 --config $(CONFIG_PHASE3)

report:
	$(PYTHON) -m src.cli report --config $(CONFIG_PHASE1)

clean:
	rm -rf .pytest_cache .mypy_cache build dist *.egg-info
