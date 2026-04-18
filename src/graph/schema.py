from __future__ import annotations

from dataclasses import dataclass

NODE_TYPE_PROJECT = "project"
NODE_TYPE_COMPANY = "company"

EDGE_ROYALTY = "royalty"
EDGE_COMMERCIAL = "commercial"
EDGE_PERFORMANCE = "performance"
EDGE_SIMILARITY = "similarity"

EDGE_TYPES = (EDGE_ROYALTY, EDGE_COMMERCIAL, EDGE_PERFORMANCE, EDGE_SIMILARITY)


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
