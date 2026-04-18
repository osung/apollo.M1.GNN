# Model Architecture

## GNN 인코더
- 대상: 이분 + heterogeneous edge type 그래프.
- 후보: R-GCN / CompGCN / HGT. 기본값 HGT (`config/model.yaml`).
- 초기 피처: `norm_embed`.
- 출력: 노드별 latent embedding `z ∈ R^d`.

## 학습 목적 — Link ranking
- 양성 간선의 `z` 쌍 내적이 음성 쌍보다 커지도록 학습.
- 간선 타입별 가중치를 loss에 반영: weighted BPR 또는 weighted contrastive.
- 평가용 held-out 간선은 그래프에서 제거한 상태로 학습(leakage 방지).

## 추론
- 쿼리 노드의 `z`와 반대편 타입 전체 노드의 `z` 간 유사도 상위 100개를 반환.
- **이미 연결된 상대도 필터링하지 않음** (rule #4).

## Cold Start — Projection MLP
- 학습 그래프에 포함된 노드에 대해 `(norm_embed, z)` 쌍을 데이터셋으로 구성.
- MLP가 `norm_embed → z_hat` 을 학습. Loss는 cosine 또는 MSE.
- 추론 분기:
  - 학습 그래프에 존재 → GNN의 `z` 사용.
  - 신규(cold) 노드 → MLP의 `z_hat` 사용.
  - 이후 동일한 FAISS 인덱스에서 top-100 검색.

## 서빙
- FAISS 인덱스 2개: `faiss_project.index`, `faiss_company.index` (inner product, z가 L2-normalized면 cosine과 동치).
- 과제 쿼리 → 기업 인덱스 조회. 기업 쿼리 → 과제 인덱스 조회.
