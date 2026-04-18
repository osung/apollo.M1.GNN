# Graph Construction

## 입력
- 과제 노드 DataFrame (`norm_embed` 포함)
- 기업 노드 DataFrame (`norm_embed` 포함)
- 3종 간선 DataFrame (royalty / commercial / performance)
- `config/graph.yaml`

## 절차
1. **노드 인덱싱**: 과제·기업 각각 `NodeMap`으로 id ↔ index 매핑 고정.
2. **노드 피처**: `norm_embed`를 float32로 캐스팅, `(N, D)` 행렬로 스택.
3. **간선 로드**: 각 간선 파일에서 `(project_id, company_id)` 쌍만 추출, 중복 제거.
4. **간선 인덱싱**: NodeMap에 존재하는 쌍만 남기고 index 공간으로 변환.
5. **Held-out split**: royalty / commercial / performance 각 관계에서 `held_out.ratio` 비율을 seed 고정하여 분리. 학습 그래프에서는 제거, 별도 파일 `data/processed/held_out.pt`로 저장.
6. **유사도 간선 (분리 캐시)**: 별도 스크립트에서 top-k 또는 top-% 유사도 간선을 계산하여 `data/processed/sim_edges.npz`로 저장. 기존 3종 간선과의 중복은 `drop_overlap_with_known_edges` 플래그에 따라 제거. Base 그래프와 분리되어 있어 top-k 파라미터 스윕 시 base 재빌드 불필요.
7. **HeteroData 빌드 (base)**: PyG `HeteroData`에 3종 실제 관계를 relation key로 저장. `edge_weights_as_attr=true`면 간선 타입 weight를 `edge_attr`로 기록. `reverse_edges.enabled=true`면 reverse 관계도 추가.
8. **저장**: `data/processed/graph.pt` (base graph). 학습 시점에 config flag로 유사도 캐시를 로드하여 merge.

## 불변식
- 과제-과제 / 기업-기업 간선은 **생성 금지**.
- 각 노드는 반드시 `project` / `company` 중 하나의 타입으로만 분류.
- 간선 우선순위(가중치)는 `royalty(1) > commercial(2) > performance(3) > similarity(4)`.

## 중복 정책
- 기본: `heterogeneous` — 동일 (과제, 기업) 쌍이 여러 관계에 나타나면 각 edge type에 각각 기록하여 관계별로 학습 가능하게 유지.
- 옵션: config에서 단일 edge로 병합 가능 (max weight 선택 등).
