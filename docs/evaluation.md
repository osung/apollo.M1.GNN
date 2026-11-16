# Evaluation Protocol

## 데이터 분할
- 각 관계(royalty / commercial / performance)에서 간선의 `held_out_ratio` (기본 10%)를 무작위 샘플링하여 테스트 셋으로 분리.
- 테스트 간선은 학습 그래프에서 제거한다 (leakage 방지).

## 지표
보고 지표는 K ∈ {10, 100} 에 대한 Hits / NDCG / Recall 6종이다 (`src/eval/ranking.py::evaluate`).

- **Hits@10 / Hits@100**: 쿼리 노드별, top-K에 held-out 상대가 **하나라도** 포함되면 1, 아니면 0. 쿼리 평균 (hit rate).
- **Recall@10 / Recall@100**: 쿼리 노드별, 반대편 타입 top-K에 실제 held-out 상대가 포함된 비율의 평균.
- **NDCG@10 / NDCG@100**: relevance = 간선 타입 가중치 (royalty > commercial > performance).

## 절차
1. Held-out 간선을 제거한 그래프로 GNN 학습.
2. 학습된 `z`로 FAISS 인덱스 빌드.
3. 각 held-out 간선 (u, v)에 대해:
   - u의 `z`로 반대편 타입 인덱스에서 top-100 검색.
   - v가 결과에 포함됐는지 여부와 순위로 Hits / NDCG / Recall 계산 (K=10, 100).
4. 간선 타입별로 분리 집계 + 전체 집계를 함께 보고.

## Cold Start 평가
- 일부 노드 자체를 학습 그래프에서 제거 → 해당 노드는 projection MLP 경로로만 평가.
- 동일한 Hits / NDCG / Recall (@10, @100) 지표를 cold / warm 세그먼트로 분리 보고.

## 검색 모델 + LLM 매칭 비교 (`config/llm_match.yaml`)
`~/work/apollo` 의 과제↔기업 매칭 방식(후보 검색 → LLM 적합도 채점 → 재랭킹)을 같은 held-out 간선으로 평가한다.
후보 검색 모델(retriever)은 임베딩 유사도(`config/paths.yaml` 의 `embedding`, 현재 bge-m3), LightGCN, HGT, LightFM 등이며, 각 모델의 단독 순위와 LLM 재랭킹 순위(`<모델>+LLM`)를 함께 보고한다.

1. 후보 검색 모델 준비 — GNN 은 `scripts/train_gnn.py` 로 학습해 z 를 저장하고 `retrievers` 에 태그를 적는다. LightFM 은 `scripts/baseline_lightfm.py --save-candidates` 로 후보 파일을 직접 쓴다.
2. `make llm-prepare` — held-out 쿼리 표본(관계별 `queries.max_per_relation`)에 대해 모델별 상위 `candidates.topn` 후보를 뽑고, 쿼리별 후보 **합집합**을 20개 단위 채점 프롬프트로 만든다 (`data/processed/llm_match/jobs_<direction>.jsonl`). (쿼리, 후보) 쌍은 한 번만 채점되고 점수는 그 쌍을 제안한 모든 모델이 공유한다.
3. `make llm-score SHARD=i NUM_SHARDS=n` — vLLM docker(`scripts/run_llm.sh`)에서 0~100 적합도 채점. 채점 프롬프트는 apollo 의 `score_01_c2p` / `score_02_p2c` 와 동일. 청크마다 저장하며 재실행 시 이어서 진행. 쿼리 순서가 무작위라 중간까지 끝난 쿼리도 무작위 표본이다.
4. `make llm-eval` — 모든 모델의 후보가 채점 완료된 **동일 쿼리 집합**에서 모델별 단독 / +LLM 의 Hits·NDCG·Recall (@10, @100) 을 관계별로 보고.

유의사항:
- LLM 은 각 모델 상위 N 후보의 순서만 바꾼다. N = 100 이면 Hits@100 / Recall@100 은 그 모델 단독과 같고, @10 과 NDCG 만 달라진다.
- 과제 설명문에 수행기관명이 적혀 있어, 과제와 간선으로 연결된 기업명은 과제 문서에서 ○○○ 로 가린다 (`texts.mask_linked_company_names`). 끄면 LLM 이 업체명 일치로 정답을 맞히는 누설이 생긴다.
- 기업설명문은 그래프 기업의 약 45% 에만 있다. 없는 기업은 업체명·산업분류·키워드만으로 채점된다.
- 이미 연결된 상대도 후보에서 제외하지 않는다 (rule #4).
