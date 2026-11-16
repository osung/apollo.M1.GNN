# Prefer the project virtualenv created by `make setup`.
PY ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

# Second machine for experiments (promaxgb10-60d4). Addressed over the direct
# 200GbE link; the hostname resolves to the much slower campus LAN address.
REMOTE ?= 192.168.100.2
REMOTE_DIR ?= work/apollo.M1.GNN
DATA_ROOT ?= $(HOME)/data/apollo.data

.PHONY: setup sync-code sync-data data merge-projects similarity hard-negatives visualize compress-graph decompress-graph train train-coldstart index eval eval-embeddings infer test lint baseline-two-tower baseline-cf baseline-lightfm llm-prepare llm-score llm-eval sync-results

setup:
	scripts/setup_env.sh $(DATA_ROOT)

# Push the working tree (code only) to the remote machine.
sync-code:
	rsync -a --exclude /.venv --exclude /.venv-bootstrap --exclude /data --exclude /outputs \
		--exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude .mypy_cache \
		./ $(REMOTE):$(REMOTE_DIR)/

# Push raw + merged + processed data (graph, similarity caches) to the remote machine.
sync-data:
	rsync -a --info=progress2 $(DATA_ROOT)/raw $(DATA_ROOT)/raw_merged $(DATA_ROOT)/processed \
		$(REMOTE):data/apollo.data/

data:
	$(PY) scripts/build_graph.py

merge-projects:
	$(PY) scripts/merge_multiyear_projects.py --paths config/paths_raw.yaml

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

# --- pro-sroberta 임베딩 + LLM 매칭 비교 (config/llm_match.yaml) ---
DIRECTION ?= p2c
SHARD ?= 0
NUM_SHARDS ?= 1
Z_TAGS ?=

llm-prepare:
	$(PY) scripts/llm_match_prepare.py

# vLLM docker 안에서 실행. 2대 분산: 로컬 `make llm-score SHARD=0 NUM_SHARDS=2`,
# 원격 `make llm-score SHARD=1 NUM_SHARDS=2` (재실행하면 이어서 채점).
llm-score:
	scripts/run_llm.sh python3 scripts/llm_match_score.py --direction $(DIRECTION) \
		--shard-id $(SHARD) --num-shards $(NUM_SHARDS)

llm-eval:
	$(PY) scripts/llm_match_eval.py --direction $(DIRECTION) $(foreach t,$(Z_TAGS),--z-tag $(t))

# 원격 장비의 LLM 채점 결과(scores_*.jsonl)를 가져온다.
sync-results:
	rsync -a --include 'scores_*.jsonl' --exclude '*' \
		$(REMOTE):data/apollo.data/processed/llm_match/ $(DATA_ROOT)/processed/llm_match/

infer:
	$(PY) scripts/infer.py --query "$(QUERY)"

test:
	pytest -q

lint:
	ruff check src scripts tests
	mypy src
