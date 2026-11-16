import numpy as np

from src.data.patents import patent_edges, split_biz_numbers
from src.graph.aux_edges import drop_pairs, inherited_edges, keyword_edges, normalize_keywords


def test_split_biz_numbers_drops_placeholders():
    assert split_biz_numbers("3148204099 ; -") == ["3148204099"]
    assert split_biz_numbers("-") == []
    assert split_biz_numbers(None) == []
    assert split_biz_numbers("111 ; 222") == ["111", "222"]


def test_patent_edges_cross_product_with_canonical_map():
    df = patent_edges(
        [["P1", "P2"], ["P3"], ["P9"], ["P1"]],
        ["111 ; 222", "-", "333", "111"],
        canonical_map={"P1": "P1", "P2": "P1", "P3": "P3"},   # P2 merged into P1, P9 unknown
    )
    assert df.values.tolist() == [["P1", "111"], ["P1", "222"]]


def test_normalize_keywords():
    assert normalize_keywords(["AI 반도체", "ai반도체", " ", None]) == {"ai반도체"}
    assert normalize_keywords(None) == set()


def test_keyword_edges_jaccard_topk_and_filters():
    projects = [{"a", "b", "c"}, {"a", "b", "x"}, {"z"}]
    companies = [{"a", "b"}, {"a", "b", "c", "d"}, {"q"}, {"a"}]
    ei, jac = keyword_edges(projects, companies, topk=1, min_shared=2)
    got = {(int(p), int(c)): round(float(j), 3) for p, c, j in zip(ei[0], ei[1], jac)}
    # project 0: company 1 (3/4=0.75) beats company 0 (2/3); project 1: company 0 (2/3)
    # company 0 -> project 0 (2/3, tie with project 1 broken by index); company 1 -> project 0
    assert got == {(0, 1): 0.75, (1, 0): 0.667, (0, 0): 0.667}
    assert all(p != 2 for p in ei[0])                 # no shared keyword -> no edge

    ei2, _ = keyword_edges(projects, companies, topk=1, min_shared=2,
                           exclude_keys=np.array([0 * 4 + 1]))
    assert (0, 1) not in set(zip(ei2[0].tolist(), ei2[1].tolist()))

    # 'a' and 'b' are too frequent once max_df=1 -> nothing reaches min_shared
    ei3, _ = keyword_edges(projects, companies, topk=1, min_shared=2, max_df=1)
    assert ei3.shape[1] == 0


def test_inherited_edges_from_similar_projects_and_companies():
    # projects 0 and 1 are near-duplicates, project 2 is different
    px = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0]], dtype=np.float32)
    # companies 0 and 1 near-duplicates, company 2 different
    cx = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]], dtype=np.float32)
    observed = np.array([[0, 2], [0, 2]])             # project0-company0, project2-company2
    ei, score = inherited_edges(px, cx, observed, k_neighbors=1, top_m=1)
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert (1, 0) in pairs        # project 1 inherits company 0 from similar project 0
    assert (0, 1) in pairs        # company 1 inherits project 0 from similar company 0
    assert all(s > 0 for s in score)

    held_out = np.array([1 * 3 + 0])                  # pretend (project1, company0) is held out
    ei2, _ = inherited_edges(px, cx, observed, k_neighbors=1, top_m=1, exclude_keys=held_out)
    assert (1, 0) not in set(zip(ei2[0].tolist(), ei2[1].tolist()))


def test_inherited_edges_never_use_a_node_as_its_own_neighbor():
    px = np.eye(3, dtype=np.float32)
    cx = np.eye(3, dtype=np.float32)
    observed = np.array([[0], [0]])
    ei, _ = inherited_edges(px, cx, observed, k_neighbors=1, top_m=1,
                            exclude_keys=np.array([0]))
    # the only warm project is 0 itself; it must not hand its own partner back to itself
    assert (0, 0) not in set(zip(ei[0].tolist(), ei[1].tolist()))


def test_drop_pairs():
    ei = np.array([[0, 1, 2], [5, 6, 7]])
    out, n = drop_pairs(ei, 10, np.array([1 * 10 + 6]))
    assert out.tolist() == [[0, 2], [5, 7]] and n == 1


def test_keyword_exclusion_with_large_index_space():
    """Pair keys exceed int32 on the real graph (430k x 392k nodes); the exclusion of
    held-out pairs must still work there."""
    n_p, n_c = 9000, 300_000                      # 8999 * 300000 > 2**31
    projects = [set() for _ in range(n_p)]
    companies = [set() for _ in range(n_c)]
    projects[8999] = {"k1", "k2"}
    companies[299_999] = {"k1", "k2"}
    companies[5] = {"k1", "k2", "k3"}
    held_out = np.array([8999 * n_c + 299_999], dtype=np.int64)
    ei, _ = keyword_edges(projects, companies, topk=2, min_shared=2, exclude_keys=held_out)
    pairs = set(zip(ei[0].tolist(), ei[1].tolist()))
    assert (8999, 299_999) not in pairs
    assert (8999, 5) in pairs
