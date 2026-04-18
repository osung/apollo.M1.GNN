# Evaluation Protocol

## 데이터 분할
- 각 관계(royalty / commercial / performance)에서 간선의 `held_out_ratio` (기본 10%)를 무작위 샘플링하여 테스트 셋으로 분리.
- 테스트 간선은 학습 그래프에서 제거한다 (leakage 방지).

## 지표
- **Recall@100**: 쿼리 노드별, 반대편 타입 top-100에 실제 held-out 상대가 포함된 비율의 평균.
- **NDCG@100**: relevance = 간선 타입 가중치 (royalty > commercial > performance).

## 절차
1. Held-out 간선을 제거한 그래프로 GNN 학습.
2. 학습된 `z`로 FAISS 인덱스 빌드.
3. 각 held-out 간선 (u, v)에 대해:
   - u의 `z`로 반대편 타입 인덱스에서 top-100 검색.
   - v가 결과에 포함됐는지 여부와 순위로 Recall / NDCG 계산.
4. 간선 타입별로 분리 집계 + 전체 집계를 함께 보고.

## Cold Start 평가
- 일부 노드 자체를 학습 그래프에서 제거 → 해당 노드는 projection MLP 경로로만 평가.
- 동일한 Recall@100 / NDCG@100 지표를 cold / warm 세그먼트로 분리 보고.
