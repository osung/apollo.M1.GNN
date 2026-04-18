# Data Schema

## 노드 파일

### 과제 (`data/raw/public_RnD_embeddings_pro_260324.pkl`)
pandas DataFrame, shape ≈ (802,710, 5)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `과제고유번호` | str | 과제 primary key |
| `과제명` | str | 과제 이름 |
| `키워드_리스트` | list[str] | 키워드 |
| `norm_embed` | np.ndarray(float32) | L2-normalized 임베딩 |
| `유망성점수` | float | 사전 계산된 유망성 점수 |

### 기업 (`data/raw/company_embeddings_pro_0925_260324.pkl`)
pandas DataFrame, shape ≈ (817,117, 8)

| 컬럼 | 타입 | 설명 |
|---|---|---|
| `사업자번호` | str | 기업 primary key |
| `한글업체명` | str | 업체명 |
| `10차산업코드명` | str | 업종 분류 |
| `ASTI 여부` | bool/str | ASTI 포함 여부 |
| `특구 여부` | bool/str | 특구 포함 여부 |
| `키워드_리스트` | list[str] | 키워드 |
| `norm_embed` | np.ndarray(float32) | L2-normalized 임베딩 |
| `유망성점수` | float | 사전 계산된 유망성 점수 |

## 간선 파일

### 기술이전 — Royalty (우선순위 1)
`data/raw/df_pr_royalty_260417_valid_biz.pkl`, shape ≈ (17,664, 9)

- project key: `과제고유번호`
- company key: `기술실시대상기관_사업자번호`
- 기타: `기술실시계약명`, `기술실시대상기관명`, `기술실시내용`, `기술료_기발생액`, `당해연도 기술료(원)`, `성과발생년도`, `과제명-국문`

### 직접 사업화 — Commercial (우선순위 2)
`data/raw/df_pr_commercial_260417_valid_biz.pkl`, shape ≈ (161,179, 12)

- project key: `과제고유번호`
- company key: `사업화주체_사업자등록번호`
- 기타: `사업화명`, `사업화내용`, `업체명`, `기매출액(원)`, `당해년도매출액(원)`, `과제명-국문`, `기준년도`, `사업화년도`, `사업화주체_업체명`, `제품명`

### 과제 수행 — Performance (우선순위 3)
`data/raw/df_project_bizno_valid.pkl`, shape ≈ (69,443, 4)

- project key: `과제고유번호`
- company key: `사업자번호`
- 기타: `과제명`, `한글업체명`

### 유사도 — Similarity (우선순위 4)
런타임 생성. `norm_embed` 코사인 유사도 top-1%.
