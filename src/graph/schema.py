from __future__ import annotations

from dataclasses import dataclass

NODE_TYPE_PROJECT = "project"
NODE_TYPE_COMPANY = "company"

EDGE_ROYALTY = "royalty"
EDGE_COMMERCIAL = "commercial"
EDGE_PERFORMANCE = "performance"
EDGE_SIMILARITY = "similarity"

EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE, EDGE_SIMILARITY)

# Held-out (evaluated) relations.
EVAL_EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE)

# A further observed project-company relation: the company is an applicant of a patent
# produced by the project. Used for training (positives + message passing) but never
# held out, so the evaluation set stays the three relations above.
EDGE_PATENT = "patent"
OBSERVED_EDGE_TYPES = (*EVAL_EDGE_TYPES, EDGE_PATENT)

# Derived project-company edges, message passing only (like similarity edges):
#   keyword   - keyword-set Jaccard between a project and a company
#   inherited - partners of the most similar same-type nodes: a project inherits the
#               companies linked to similar projects, a company the projects linked to
#               similar companies. The result is still a project-company edge, so the
#               bipartite invariant holds (no project-project / company-company edge).
EDGE_KEYWORD = "keyword"
EDGE_INHERITED = "inherited"
AUX_EDGE_TYPES = (EDGE_SIMILARITY, EDGE_KEYWORD, EDGE_INHERITED)


def pair_keys(edge_index, n_company: int):
    """(2, E) project/company index pairs -> one integer key per pair."""
    return edge_index[0] * n_company + edge_index[1]


@dataclass(frozen=True)
class NodeMap:
    id_to_idx: dict
    idx_to_id: list

    @classmethod
    def from_ids(cls, ids) -> "NodeMap":
        idx_to_id = list(ids)
        id_to_idx = {v: i for i, v in enumerate(idx_to_id)}
        return cls(id_to_idx=id_to_idx, idx_to_id=idx_to_id)

    def __len__(self) -> int:
        return len(self.idx_to_id)
