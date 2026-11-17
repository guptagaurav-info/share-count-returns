PY ?= python
export PYTHONPATH := src

.PHONY: all ingest run test
all: ingest run
ingest:
	$(PY) -m scr.ingest
run:
	$(PY) -m scr.pipeline
test:
	$(PY) -m pytest -q tests
