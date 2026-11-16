"""Project-company edges from the patent table.

Each patent row lists the national R&D projects it came from (`과제번호`, a list)
and its applicants' business registration numbers (`출원기관사업자번호`, ';'-separated,
'-' when unknown). A (project, applicant company) pair is an observed relation:
the company filed a patent that is an outcome of the project.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd


def _as_list(value: object) -> list[str]:
    if value is None or isinstance(value, float):
        return []
    if isinstance(value, str):
        return [value]
    try:
        return [str(v) for v in value]  # type: ignore[union-attr]
    except TypeError:
        return []


def split_biz_numbers(value: object, sep: str = ";") -> list[str]:
    """'3148204099 ; -' -> ['3148204099'] (blank and '-' placeholders dropped)."""
    out = []
    for part in str(value if value is not None else "").split(sep):
        part = part.strip()
        if part and part != "-" and part.lower() != "nan":
            out.append(part)
    return out


def patent_edges(
    project_lists: Iterable[object],
    applicant_fields: Iterable[object],
    canonical_map: Mapping[str, str] | None = None,
    sep: str = ";",
) -> pd.DataFrame:
    """Unique (project_id, company_id) pairs, one per project x applicant of a patent.

    canonical_map: raw project id -> canonical id of the multi-year merge. Projects
    missing from the map are dropped (the graph builder would drop them anyway).
    """
    rows: set[tuple[str, str]] = set()
    for projects, applicants in zip(project_lists, applicant_fields, strict=True):
        companies = split_biz_numbers(applicants, sep)
        if not companies:
            continue
        for pid in _as_list(projects):
            pid = pid.strip()
            if canonical_map is not None:
                pid = canonical_map.get(pid, "")
            if not pid:
                continue
            rows.update((pid, c) for c in companies)
    return pd.DataFrame(sorted(rows), columns=["project_id", "company_id"])
