import json

import numpy as np

from scripts.llm_match_score import MockEngine, score_chunk
from src.baselines.llm_match import (
    core_name,
    extend_candidates,
    iter_jobs,
    load_candidates,
    load_scores,
    mask_names,
    rerank,
    save_candidates,
    select_queries,
    tiered_candidates,
    topn_candidates,
    union_candidates,
)
from src.baselines.llm_prompt import (
    build_score_messages,
    company_doc_text,
    parse_scores,
    project_doc_text,
    strip_think,
)


def test_doc_text_omits_missing_fields():
    company = {"한글업체명": "A", "키워드_리스트": ["x", "y"]}
    assert company_doc_text(company) == "업체명: A\n키워드: x, y"
    text = project_doc_text({"과제명": "P", "제출년도": 2020, "과제설명문": float("nan")})
    assert text == "과제명: P\n제출년도: 2020"


def test_build_score_messages_direction():
    p2c = build_score_messages("S", "국가 R&D 과제", [(1, "a"), (2, "b")], "기업", tech_side="source")
    assert "[국가 R&D 과제]\nS" in p2c[1]["content"]
    assert "1. a\n2. b" in p2c[1]["content"]
    assert "기준 문서의 기술 → 후보 기업의 개선" in p2c[1]["content"]
    c2p = build_score_messages("S", "기업", [(1, "a")], "국가 R&D 과제", tech_side="candidate")
    assert "후보의 기술 → 기준 문서의 기업 개선" in c2p[1]["content"]


def test_parse_scores_json_and_truncated():
    ok = parse_scores('noise {"results": [{"id": 1, "score": 80, "reason": "r"}]} tail')
    assert ok == {1: (80, "r")}
    cut = '{"results": [{"id": 1, "score": 80, "reason": "r"}, {"id": 2, "score": 30, "reason": "q"}, {"id": 3, "sc'
    assert parse_scores(cut) == {1: (80, "r"), 2: (30, "q")}
    assert parse_scores('{"results": [{"id": 1, "score": 300, "reason": ""}]}')[1][0] == 100


def test_strip_think():
    assert strip_think("<think>abc</think>\n{}") == "{}"


def test_rerank_llm_first_then_embedding():
    cand = np.array([10, 11, 12, 13])
    base = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    llm = np.array([50, 90, 50, 0], dtype=np.float32)
    assert rerank(cand, base, llm).tolist() == [11, 10, 12, 13]


def test_select_queries_sampled_and_deterministic():
    per_rel = {"a": list(range(100)), "b": [1, 2, 3]}
    q1 = select_queries(per_rel, 10, seed=0)
    q2 = select_queries(per_rel, 10, seed=0)
    assert q1.tolist() == q2.tolist()
    assert q1.tolist() != sorted(q1.tolist())  # random order, not sorted
    assert {1, 2, 3} <= set(q1.tolist())
    assert len(q1) <= 13
    assert len(select_queries(per_rel, None, seed=0)) == 100


def test_topn_candidates_cosine_order():
    cand = np.eye(4, dtype=np.float32)
    q = np.array([[0.1, 0.9, 0.3, 0.0]], dtype=np.float32)
    idx, score = topn_candidates(q, cand, 3)
    assert idx.tolist() == [[1, 2, 0]]
    assert score[0, 0] == np.float32(0.9)


def test_jobs_score_and_load_roundtrip(tmp_path):
    q = np.array([5, 7])
    cands = np.array([[0, 1, 2], [2, 3, 4]])
    p_ids = [f"p{i}" for i in range(8)]
    c_ids = [f"c{i}" for i in range(5)]
    p_text = {f"p{i}": {"과제명": f"과제{i}"} for i in range(8)}
    c_text = {f"c{i}": {"한글업체명": f"기업{i}"} for i in range(5)}
    jobs = list(iter_jobs("p2c", q, cands, p_ids, c_ids, p_text, c_text, batch=2))
    assert [j["job_id"] for j in jobs] == ["p2c:5:0", "p2c:5:1", "p2c:7:0", "p2c:7:1"]
    assert jobs[0]["src_text"] == "과제명: 과제5"
    assert jobs[0]["cand_texts"] == ["업체명: 기업0", "업체명: 기업1"]
    assert jobs[2]["query_rank"] == 1

    records, _, _ = score_chunk(MockEngine(), jobs, retry_batch=10)
    assert all(not r["missing"] for r in records)
    path = tmp_path / "scores_p2c_shard0of1.jsonl"
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n{broken")
    scores = load_scores([path])["p2c"]
    assert set(scores[5]) == {0, 1, 2}
    assert set(scores[7]) == {2, 3, 4}


class _HalfEngine:
    """Answers only the first candidate of each prompt."""

    def generate(self, messages_list):
        return ['{"results": [{"id": 1, "score": 70, "reason": "x"}]}'] * len(messages_list), 0, 0


def test_score_chunk_marks_unanswered_as_missing():
    job = {"job_id": "p2c:1:0", "direction": "p2c", "query": 1, "cands": [4, 5, 6],
           "src_label": "국가 R&D 과제", "cand_label": "기업", "tech_side": "source",
           "src_text": "s", "cand_texts": ["a", "b", "c"]}
    (rec,), _, _ = score_chunk(_HalfEngine(), [job], retry_batch=10)
    # round 1 answers cand 4; the retry prompt holds [5, 6] and answers its first (5)
    assert rec["scores"]["4"] == [70, "x"]
    assert rec["scores"]["5"] == [70, "x"]
    assert rec["scores"]["6"][0] == 0
    assert rec["missing"] == [6]


def test_mask_names_hides_company_in_project_text():
    assert core_name("(주) 썬 커뮤니케이션") == "썬커뮤니케이션"
    text = "(주)썬커뮤니케이션에서 수행하는 이 과제는 썬 커뮤니케이션의 기술이다."
    assert mask_names(text, ["주식회사 썬커뮤니케이션", "가"]) == "(주)○○○에서 수행하는 이 과제는 ○○○의 기술이다."


def test_iter_jobs_masks_by_direction():
    p_text = {"p0": {"과제명": "가나전자 공동개발"}}
    c_text = {"c0": {"한글업체명": "(주)가나전자"}}
    q, cands = np.array([0]), np.array([[0]])
    (job,) = iter_jobs("p2c", q, cands, ["p0"], ["c0"], p_text, c_text, 20, mask={0: ["(주)가나전자"]})
    assert job["src_text"] == "과제명: ○○○ 공동개발"
    assert job["cand_texts"] == ["업체명: (주)가나전자"]
    (job,) = iter_jobs("c2p", q, cands, ["p0"], ["c0"], p_text, c_text, 20, mask={0: ["(주)가나전자"]})
    assert job["src_text"] == "업체명: (주)가나전자"
    assert job["cand_texts"] == ["과제명: ○○○ 공동개발"]


def test_union_candidates_dedups_per_query():
    a = np.array([[1, 2, 3], [7, 8, 9]])
    b = np.array([[3, 4, 1], [9, 10, 11]])
    u = union_candidates([a, b], seed=0)
    assert sorted(u[0]) == [1, 2, 3, 4]
    assert sorted(u[1]) == [7, 8, 9, 10, 11]
    assert u == union_candidates([a, b], seed=0)


def test_candidates_roundtrip_by_retriever(tmp_path):
    q = np.array([3, 1])
    save_candidates(tmp_path, "p2c", "lightfm", q, np.array([[1, 2], [3, 4]]), np.ones((2, 2)))
    save_candidates(tmp_path, "p2c", "bge-m3", q, np.array([[5, 6], [7, 8]]), np.ones((2, 2)))
    save_candidates(tmp_path, "c2p", "hgt", q, np.array([[0, 0], [0, 0]]), np.ones((2, 2)))
    got = load_candidates(tmp_path, "p2c")
    assert sorted(got) == ["bge-m3", "lightfm"]
    assert got["lightfm"]["cand_idx"].tolist() == [[1, 2], [3, 4]]


def test_tiered_candidates_first_tier_matches_union_and_tiers_are_disjoint():
    rng = np.random.default_rng(0)
    a = np.stack([rng.permutation(50)[:12] for _ in range(4)])
    b = np.stack([rng.permutation(50)[:12] for _ in range(4)])
    tiers = tiered_candidates([a, b], [4, 8, 12], seed=3)
    assert tiers[0] == union_candidates([a[:, :4], b[:, :4]], seed=3)
    for q in range(4):
        rows = [t[q] for t in tiers]
        flat = [c for r in rows for c in r]
        assert len(flat) == len(set(flat))
        assert set(rows[0]) | set(rows[1]) == set(a[q, :8]) | set(b[q, :8])
        assert set(flat) == set(a[q]) | set(b[q])


def test_iter_jobs_continues_job_numbers_across_depth_tiers():
    q = np.array([0])
    p_text = {"p0": {"과제명": "과제"}}
    c_text = {f"c{i}": {"한글업체명": f"기업{i}"} for i in range(6)}
    c_ids = [f"c{i}" for i in range(6)]
    first = list(iter_jobs("p2c", q, [[0, 1, 2]], ["p0"], c_ids, p_text, c_text, 2, depth=3))
    more = list(iter_jobs("p2c", q, [[3, 4, 5]], ["p0"], c_ids, p_text, c_text, 2,
                          first_batch=[len(first)], depth=6))
    assert [j["job_id"] for j in first] == ["p2c:0:0", "p2c:0:1"]
    assert [j["job_id"] for j in more] == ["p2c:0:2", "p2c:0:3"]
    assert {j["depth"] for j in more} == {6}


def test_extend_candidates_keeps_saved_prefix():
    prev = np.array([[5, 3]])
    prev_s = np.array([[0.9, 0.8]], dtype=np.float32)
    new = np.array([[3, 5, 7, 9]])          # same top-2 in another order, then deeper ones
    new_s = np.array([[0.9, 0.9, 0.7, 0.6]], dtype=np.float32)
    idx, score = extend_candidates(prev, prev_s, new, new_s)
    assert idx.tolist() == [[5, 3, 7, 9]]
    assert np.allclose(score, [[0.9, 0.8, 0.7, 0.6]])
