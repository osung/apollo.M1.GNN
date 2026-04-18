PY ?= python

.PHONY: data compress-graph decompress-graph train train-coldstart index eval infer test lint baseline-two-tower baseline-cf baseline-lightfm

data:
	$(PY) scripts/build_graph.py

compress-graph:
	$(PY) scripts/compress_graph.py

decompress-graph:
	$(PY) scripts/compress_graph.py --mode decompress --input data/processed/graph_fp16.pt

train:
	$(PY) scripts/train_gnn.py

train-coldstart:
	$(PY) scripts/train_coldstart.py

index:
	$(PY) scripts/build_index.py

eval:
	$(PY) scripts/evaluate.py

baseline-two-tower:
	$(PY) scripts/baseline_two_tower.py

baseline-cf:
	$(PY) scripts/baseline_cf.py

baseline-lightfm:
	$(PY) scripts/baseline_lightfm.py

infer:
	$(PY) scripts/infer.py --query "$(QUERY)"

test:
	pytest -q

lint:
	ruff check src scripts tests
	mypy src
