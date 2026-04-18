"""Zero-training two-tower baseline.

Both sides already live in a shared latent space as L2-normalized `norm_embed`
vectors (i.e. the dual-encoder was trained upstream). Inner product is
therefore cosine similarity, and the only thing this "model" does at serve
time is a FAISS top-K lookup against the opposite-type index.

Use this as a lower bound that uses **no interaction signal whatsoever** —
any trained model (LightFM, GNN, CF) should beat it meaningfully; if it
doesn't, the interaction data is not buying generalization.
"""
from __future__ import annotations

import faiss
import numpy as np


class TwoTowerBaseline:
    def __init__(self, project_x: np.ndarray, company_x: np.ndarray):
        if project_x.shape[1] != company_x.shape[1]:
            raise ValueError(
                f"embed dim mismatch: project {project_x.shape[1]} vs "
                f"company {company_x.shape[1]}"
            )
        self.project_x = np.ascontiguousarray(project_x.astype(np.float32))
        self.company_x = np.ascontiguousarray(company_x.astype(np.float32))
        d = self.project_x.shape[1]

        self._project_index = faiss.IndexFlatIP(d)
        self._project_index.add(self.project_x)
        self._company_index = faiss.IndexFlatIP(d)
        self._company_index.add(self.company_x)

    def recommend_companies(
        self, project_indices: np.ndarray, topk: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._search(
            query_x=self.project_x,
            query_ids=np.asarray(project_indices, dtype=np.int64),
            index=self._company_index,
            topk=topk,
        )

    def recommend_projects(
        self, company_indices: np.ndarray, topk: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._search(
            query_x=self.company_x,
            query_ids=np.asarray(company_indices, dtype=np.int64),
            index=self._project_index,
            topk=topk,
        )

    @staticmethod
    def _search(
        query_x: np.ndarray, query_ids: np.ndarray, index: faiss.Index, topk: int
    ) -> tuple[np.ndarray, np.ndarray]:
        queries = np.ascontiguousarray(query_x[query_ids])
        scores, indices = index.search(queries, topk)
        return indices.astype(np.int64), scores
