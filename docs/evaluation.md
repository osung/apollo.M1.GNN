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

## pro-sroberta 임베딩 + LLM 매칭 비교 (`config/llm_match.yaml`)
`~/work/apollo` 의 과제↔기업 매칭 방식(임베딩 후보 검색 → LLM 적합도 채점 → 재랭킹)을 같은 held-out 간선으로 평가한다.

1. `make llm-prepare` — held-out 쿼리 표본(관계별 `queries.max_per_relation`)에 대해 `norm_embed` 유사도 상위 `candidates.topn` 후보를 뽑고, 후보 20개 단위의 채점 프롬프트를 만든다 (`data/processed/llm_match/jobs_<direction>.jsonl`).
2. `make llm-score SHARD=i NUM_SHARDS=n` — vLLM docker(`scripts/run_llm.sh`)에서 0~100 적합도 채점. 채점 프롬프트는 apollo 의 `score_01_c2p` / `score_02_p2c` 와 동일. 청크마다 저장하며 재실행 시 이어서 진행.
3. `make llm-eval Z_TAGS="lightgcn_h128_l2 hgt_h128_l2_nh4"` — LLM 채점이 끝난 **동일 쿼리 집합**에서 pro-sroberta / pro-sroberta+LLM / 지정한 GNN 임베딩의 Hits·NDCG·Recall (@10, @100) 을 관계별로 보고.

유의사항:
- LLM 은 임베딩 상위 N 후보의 순서만 바꾼다. N = 100 이면 Hits@100 / Recall@100 은 pro-sroberta 단독과 같고, @10 과 NDCG 만 달라진다.
- 과제 설명문에 수행기관명이 적혀 있어, 과제와 간선으로 연결된 기업명은 과제 문서에서 ○○○ 로 가린다 (`texts.mask_linked_company_names`). 끄면 LLM 이 업체명 일치로 정답을 맞히는 누설이 생긴다.
- 기업설명문은 그래프 기업의 약 45% 에만 있다. 없는 기업은 업체명·산업분류·키워드만으로 채점된다.
- 이미 연결된 상대도 후보에서 제외하지 않는다 (rule #4).
