import numpy as np
import pytest

from src.eval.ranking import evaluate, group_ground_truth, ndcg_at_k, recall_at_k


def test_group_ground_truth_p2c():
    ei = np.array([[0, 0, 1], [10, 11, 20]], dtype=np.int64)
    gt = group_ground_truth(ei, direction="project_to_company")
    assert gt == {0: {10, 11}, 1: {20}}


def test_group_ground_truth_c2p():
    ei = np.array([[0, 0, 1], [10, 11, 20]], dtype=np.int64)
    gt = group_ground_truth(ei, direction="company_to_project")
    assert gt == {10: {0}, 11: {0}, 20: {1}}


def test_group_empty():
    ei = np.zeros((2, 0), dtype=np.int64)
    assert group_ground_truth(ei) == {}


def test_recall_perfect():
    preds = np.array([[10, 99, 88]])
    gt = {0: {10}}
    assert recall_at_k(preds, gt, [0]) == 1.0


def test_recall_miss():
    preds = np.array([[99, 88, 77]])
    gt = {0: {10}}
    assert recall_at_k(preds, gt, [0]) == 0.0


def test_recall_partial():
    preds = np.array([[10, 99, 88]])
    gt = {0: {10, 20}}
    assert recall_at_k(preds, gt, [0]) == 0.5


def test_recall_skips_queries_with_no_gt():
    preds = np.array([[10, 99], [7, 8]])
    gt = {0: {10}}
    assert recall_at_k(preds, gt, [0, 1]) == 1.0


def test_ndcg_ideal_ordering():
    preds = np.array([[10, 99, 88]])
    gt = {0: {10}}
    assert ndcg_at_k(preds, gt, [0]) == pytest.approx(1.0)


def test_ndcg_lower_when_rank_worse():
    top = ndcg_at_k(np.array([[10, 99, 88]]), {0: {10}}, [0])
    bottom = ndcg_at_k(np.array([[88, 99, 10]]), {0: {10}}, [0])
    assert top > bottom


def test_evaluate_bundle():
    preds = np.array([[10, 99, 88, 77]])
    gt = {0: {10, 77}}
    m = evaluate(preds, [0], gt, ks=(1, 4))
    assert m["recall@1"] == 0.5
    assert m["recall@4"] == 1.0
    assert 0 < m["ndcg@4"] <= 1.0


def test_evaluate_rejects_too_small():
    with pytest.raises(ValueError):
        evaluate(np.zeros((1, 3), dtype=np.int64), [0], {0: {1}}, ks=(5,))
