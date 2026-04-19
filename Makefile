PY ?= python

.PHONY: data merge-projects similarity hard-negatives visualize compress-graph decompress-graph train train-coldstart index eval eval-embeddings infer test lint baseline-two-tower baseline-cf baseline-lightfm

data:
	$(PY) scripts/build_graph.py

merge-projects:
	$(PY) scripts/merge_multiyear_projects.py

similarity:
	$(PY) scripts/build_similarity.py

hard-negatives:
	$(PY) scripts/build_hard_negatives.py

visualize:
	$(PY) scripts/visualize_graph.py

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

eval-embeddings:
	$(PY) scripts/evaluate_embeddings.py --sweep-dir data/processed/checkpoints

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
