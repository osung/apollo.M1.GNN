PY ?= python

.PHONY: data train train-coldstart index eval infer test lint

data:
	$(PY) scripts/build_graph.py

train:
	$(PY) scripts/train_gnn.py

train-coldstart:
	$(PY) scripts/train_coldstart.py

index:
	$(PY) scripts/build_index.py

eval:
	$(PY) scripts/evaluate.py

infer:
	$(PY) scripts/infer.py --query "$(QUERY)"

test:
	pytest -q

lint:
	ruff check src scripts tests
	mypy src
