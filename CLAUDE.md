# 국가 R&D 과제 - 기업 매칭 시스템

## Overview
국가 R&D 과제와 수요 기업을 연결하는 GNN 기반 추천 시스템.
과제·기업의 **이분 그래프(bipartite graph)** 를 구성하고, 과제-기업 간 네 가지
관계(기술이전 / 직접사업화 / 과제수행 / 임베딩 유사도)를 가중치가 다른 간선으로
학습하여, 임의의 노드에 대해 반대편 타입 노드 중 **가장 가까운 top-100** 을 반환한다.

학습 그래프에 포함되지 않은 신규 노드(**cold start**)에 대해서는, 사전 계산된
`norm_embed` 벡터를 학습된 latent space로 매핑하는 **projection MLP** 를 별도
학습하여 동일한 검색 파이프라인으로 추론한다.

## Tech Stack
- Language: Python 3.10+
- GNN: PyTorch 2.x, PyTorch Geometric (PyG)
- Retrieval serving: FAISS
- Data: pandas, numpy, pickle
- Config: YAML 기반 (hydra 또는 단순 `yaml.safe_load`)

## Project Structure
- `src/`
  - `data/` — pickle 로드, 그래프 객체 생성 로직
  - `graph/` — 간선 구성, 노드/엣지 스키마 정의
  - `models/` — GNN 인코더, projection MLP 정의
  - `training/` — 학습 루프, 손실 함수, 샘플러
  - `serving/` — FAISS 인덱스 빌드 및 top-k 검색
  - `eval/` — Recall@K, NDCG@K 등 평가 지표
- `data/raw/` — 제공받은 원본 pickle 파일 (git-ignored)
- `data/processed/` — 전처리된 그래프 객체, 임베딩 캐시
- `config/` — YAML 설정 (경로, 하이퍼파라미터, 가중치)
- `scripts/` — 일회성 실행 엔트리포인트
- `tests/` — pytest
- `docs/` — 상세 설계 문서 (lazy import 대상)

## Data

### 노드 데이터
| 타입 | 파일 | 비고 |
|---|---|---|
| 과제 | `data/raw/public_RnD_embeddings_pro_260324.pkl` | `norm_embed` 컬럼 보유 |
| 기업 | `data/raw/company_embeddings_pro_0925_260324.pkl` | `norm_embed` 컬럼 보유 |

`norm_embed`는 과제·기업이 **공통 latent space에 L2-normalized 상태로 사전 임베딩된**
벡터이며, 두 벡터 간 코사인 유사도가 클수록 특성이 가깝다.

### 간선 데이터 (과제 ↔ 기업, 가중치는 1이 가장 강함)
| 우선순위 | 관계 | 파일 | 의미 |
|---|---|---|---|
| 1 | 기술이전 (Royalty) | `data/raw/df_pr_royalty_260417_valid_biz.pkl` | 과제 성과가 기업에 기술이전된 관계 |
| 2 | 직접 사업화 (Commercial) | `data/raw/df_pr_commercial_260417_valid_biz.pkl` | 과제 기반 창업/사업화 관계 |
| 3 | 과제 수행 (Performance) | `data/raw/df_project_bizno_valid.pkl` | 해당 기업이 과제를 수행한 관계 |
| 4 | 임베딩 유사도 (Similarity) | (런타임 생성) | `norm_embed` 코사인 유사도 상위 1% 매칭 |

### 그래프 구성 규칙
- **이분 그래프 불변식**: 과제 노드과 기업 노드만 존재. 과제-과제, 기업-기업 간선은 **절대 생성하지 않는다.**
- **간선 타입별 가중치**: 관계 종류별 가중치는 `config/graph.yaml`에서 관리. 1 > 2 > 3 > 4 순서를 뒤집지 않는다.
- **중복 간선 처리**: 동일한 (과제, 기업) 쌍이 여러 관계에 등장할 수 있음. 기본 정책은 **heterogeneous edge type으로 유지** 하여 관계별 학습 가능하게 한다. 병합 옵션은 config에서 선택.
- **유사도 간선 구성**: 각 과제에 대해 유사도 top-1% 기업, 각 기업에 대해 유사도 top-1% 과제를 각각 뽑아 union한다. 기존 관계 간선과 중복되는 것은 유사도 간선에서 제거 가능 (config flag).

## Model Architecture

### GNN 인코더
- Bipartite + heterogeneous edge type을 처리할 수 있는 구조 사용 (예: R-GCN, CompGCN, HGT 등)
- 초기 노드 피처로 `norm_embed`를 사용
- 출력: 각 노드의 latent embedding `z`

### 학습 목적 (Link ranking)
- 양성 간선(실제 연결)은 `z` 공간에서 가깝게, 음성 샘플은 멀게 학습
- 간선 타입 가중치는 손실에 반영 (예: weighted BPR, weighted contrastive)
- 추론 시 대상 노드의 `z`와 **반대편 타입 전체 노드**의 `z` 간 유사도를 계산 → top-100 반환
- **현재 간선이 이미 존재하는 상대도 결과에서 필터링하지 않는다** (목표 정의에 따름)

### Cold Start — Projection MLP
- GNN 학습 완료 후, 학습 그래프에 포함된 노드들의
  `(norm_embed, z)` 쌍을 학습 데이터로 사용
- `norm_embed → z` 를 학습하는 MLP를 별도 학습 (MSE 또는 cosine 기반 loss)
- 추론 경로 분기:
  - **기존 노드**: GNN이 생성한 `z` 직접 사용
  - **신규 노드**: `norm_embed`를 projection MLP에 통과시켜 `z_hat` 생성
  - 두 경로 모두 동일한 FAISS 인덱스에서 top-100 검색

### 서빙
- 반대편 타입 노드 임베딩으로 FAISS 인덱스 두 개를 빌드 (과제 인덱스, 기업 인덱스)
- 과제 쿼리 → 기업 인덱스 검색, 기업 쿼리 → 과제 인덱스 검색

## Commands
- `make data` — 원본 pickle 로드, 그래프 객체 생성 (`data/processed/graph.pt`)
- `make train` — GNN 인코더 학습
- `make train-coldstart` — projection MLP 학습 (GNN 학습 완료 후 실행)
- `make index` — 학습된 임베딩으로 FAISS 인덱스 빌드
- `make eval` — held-out 간선에 대한 Recall@100 / NDCG@100
- `make infer QUERY=<node_id>` — 특정 노드에 대한 top-100 추천 출력
- `make test` — pytest
- `make lint` — ruff + mypy

## Coding Conventions
- PEP 8 준수, ruff로 자동 포맷, 타입 힌트 필수
- 경로·하이퍼파라미터 하드코딩 금지 — 전부 `config/*.yaml`에서 로드
- 시드 고정 및 모든 하이퍼파라미터 실험 로그에 기록 (재현성)
- 대용량 pickle은 필요한 컬럼만 로드, `norm_embed`는 `float32`로 캐스팅
- ETL 로직은 스크립트가 아니라 `src/data/`의 함수로 작성 → 단위 테스트 가능하게 유지
- 커밋 메시지: Conventional Commits (feat / fix / data / model / chore …)

## Rules (절대 규칙)
1. **이분 그래프 불변식**: 과제-과제, 기업-기업 간선을 생성하는 코드는 작성하지 않는다.
2. **노드 타입 식별**: 노드는 반드시 타입 필드(또는 `project_*` / `company_*` ID 규칙)로 구분한다.
3. **간선 가중치 순서**: 기술이전(1) > 사업화(2) > 수행(3) > 유사도(4). 이 순서는 config로만 조정한다.
4. **Top-100 필터링 금지**: 추론 결과에서 "이미 연결된 상대"를 제외하지 않는다.
5. **Cold start 분기**: 추론 전에 노드가 학습 그래프에 존재하는지 확인하고 GNN / MLP 경로를 분기한다.
6. **Leakage 방지**: 평가용 held-out 간선은 그래프에서 제거한 상태로 학습한다.

## Domain Terminology
- **과제 (Project)**: 국가 R&D 과제. NTIS 기반 연구개발 과제 레코드.
- **기업 (Company)**: 사업자번호 기준 기업 노드.
- **기술이전 (Royalty)**: 과제 성과가 기업에 이전되어 로열티가 발생한 관계. 가장 강한 신호.
- **사업화 (Commercial)**: 과제를 기반으로 창업 또는 제품·서비스로 직접 사업화한 관계.
- **수행 (Performance)**: 해당 기업이 과제를 수행한 관계.
- **norm_embed**: 과제·기업의 특성을 공통 latent space에 사전 임베딩한 L2-normalized 벡터.
- **z (latent embedding)**: GNN 인코더가 출력하는 노드 임베딩. top-100 검색의 기준.
- **Cold start 노드**: 학습 시점에 그래프에 포함되지 않았던 신규 노드.

## References
- 데이터 스키마 상세: @docs/data_schema.md
- 그래프 구성 파이프라인: @docs/graph_construction.md
- 모델 아키텍처 상세: @docs/model.md
- 평가 프로토콜: @docs/evaluation.md
