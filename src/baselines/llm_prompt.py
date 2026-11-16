"""LLM fit-scoring prompts and parsing for the embedding + LLM matching baseline.

Ported from the APOLLO matching pipeline (`matching_viewer_explain.py`,
prompts `score_01_c2p` / `score_02_p2c`) so that this repository scores
candidates exactly the way that pipeline does. The prompt strings below are
copied verbatim; do not reword them without re-running the comparison.

This module must stay dependency-free (stdlib only): it is imported inside
the vLLM docker image, which has none of the project's other requirements.
"""
# ruff: noqa: E501
from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

CAND_MAXLEN = 700   # per-candidate document truncation inside a prompt
FIELD_MAXLEN = 700  # truncation of the description field when building a document
MAX_KEYWORDS = 12
MISSING = (0, "(LLM 응답 누락)")

SYSTEM_PROMPT = ("당신은 기술사업화 매칭을 평가하는 전문가입니다. 기준이 되는 문서"
            "(기업·국가 R&D 과제·기술 수요·특허 등)와 후보 문서들 사이의 실질적 적합도를 "
            "냉정하게 평가합니다.")

GUIDE_CANDIDATE_IS_TECH = (
    "각 [{cand_label}] 후보의 기술·성과가 위 [{src_label}]의 제품·공정·사업을 개선하거나 기술 수요를 충족하는 데 "
    "얼마나 기여할 수 있는지 0~100 정수로 평가하세요.\n"
    "평가 방향은 항상 '후보의 기술 → 기준 문서의 기업 개선' 한 방향입니다. 기준 문서의 기업이 이미 보유한 기술·제품·"
    "장비로 후보를 개선·보완·검증·측정할 수 있다는 식의 반대 방향 관련성은 점수에 반영하지 마세요 — 그것은 매칭의 "
    "목적이 아닙니다.\n"
    "  90-100: 후보 기술이 기업의 핵심 제품·공정·수요를 직접 개선·충족\n"
    "  70-89 : 주요 부분을 개선하거나 상당 부분 충족\n"
    "  40-69 : 일부 요소만 개선에 기여하거나, 적용 대상은 다르지만 핵심 기법·소재·설계 원리를 기업의 수요기술 개발에 "
    "전이할 수 있음(부분 적합)\n"
    "  10-39 : 분야만 유사하고 개선 기여나 기법 전이가 불명확\n"
    "  0-9   : 무관하거나, 관련성이 반대 방향(기업 기술이 후보에 기여)뿐임\n"
    "기법 전이 가능성은 인정하되 직접 개선보다 낮게 두세요(같은 문제를 푸는 다른 응용, 같은 소재·구조 원리의 다른 대상 등). "
    "키워드 표면 일치나 같은 업종이라는 사실만으로 점수를 올리지 마세요. 후보 기술이 기업의 어느 제품·공정을 어떻게 "
    "바꾸는지, 또는 어떤 기법이 옮겨질 수 있는지가 근거여야 합니다.\n"
    "반드시 아래 JSON 형식으로만, 모든 후보에 대해 답하세요(reason 은 40자 이내 한국어, '후보 기술 → 무엇을 개선' 형태):\n"
    '{"results": [{"id": <후보번호>, "score": <0-100 정수>, "reason": "<근거>"}, ...]}'
)
GUIDE_SOURCE_IS_TECH = (
    "각 [{cand_label}] 후보가 위 [{src_label}]의 기술·성과를 도입·활용하여 자사 제품·공정·사업을 개선할 수 있는지 — "
    "즉 이 기술의 수요자·사업화 주체로 얼마나 적합한지 — 0~100 정수로 평가하세요.\n"
    "평가 방향은 항상 '기준 문서의 기술 → 후보 기업의 개선' 한 방향입니다. 후보 기업이 이미 보유한 기술·제품·장비로 "
    "기준 문서의 기술을 개선·보완·검증할 수 있다는 식의 반대 방향 관련성은 점수에 반영하지 마세요.\n"
    "  90-100: 후보 기업의 핵심 제품·공정이 이 기술로 직접 개선되고 도입 수요가 명확\n"
    "  70-89 : 주요 부분이 개선될 수 있음\n"
    "  40-69 : 일부 요소만 기여하거나, 적용 대상은 다르지만 이 기술의 핵심 기법·소재·설계 원리를 후보 기업이 자사 "
    "제품·공정에 전이할 수 있음(부분 적합)\n"
    "  10-39 : 분야만 유사하고 도입 효과나 기법 전이가 불명확\n"
    "  0-9   : 무관하거나, 관련성이 반대 방향뿐임\n"
    "기법 전이 가능성은 인정하되 직접 개선보다 낮게 두세요. 키워드 표면 일치나 같은 업종이라는 사실만으로 점수를 올리지 "
    "마세요. 이 기술이 후보 기업의 어느 제품·공정을 어떻게 바꾸는지, 또는 어떤 기법이 옮겨질 수 있는지가 근거여야 합니다.\n"
    "반드시 아래 JSON 형식으로만, 모든 후보에 대해 답하세요(reason 은 40자 이내 한국어, '기술 → 무엇을 개선' 형태):\n"
    '{"results": [{"id": <후보번호>, "score": <0-100 정수>, "reason": "<근거>"}, ...]}'
)


def _safe_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and v != v:  # NaN
        return ""
    s = str(v).strip()
    return "" if s.lower() in ("nan", "none", "<na>") else s


def _str_list(v: Any) -> list[str]:
    if v is None or isinstance(v, (str, bytes, float)):
        s = _safe_str(v)
        return [t.strip() for t in re.split(r"[;,]", s) if t.strip()] if s else []
    return [s for s in (_safe_str(x) for x in v) if s]


def clip_text(v: Any, n: int) -> str:
    s = _safe_str(v)
    return s[:n] if len(s) > n else s


def company_doc_text(row: Mapping[str, Any]) -> str:
    """Company document; fields without a value are omitted line by line."""
    parts = [f"업체명: {_safe_str(row.get('한글업체명'))}"]
    ind = _safe_str(row.get("10차산업코드명"))
    if ind:
        parts.append(f"산업분류: {ind}")
    kw = ", ".join(_str_list(row.get("키워드_리스트"))[:MAX_KEYWORDS])
    if kw:
        parts.append(f"키워드: {kw}")
    desc = clip_text(row.get("기업설명문"), FIELD_MAXLEN)
    if desc:
        parts.append(f"기업설명: {desc}")
    return "\n".join(parts)


def project_doc_text(row: Mapping[str, Any]) -> str:
    """Project document; fields without a value are omitted line by line."""
    parts = [f"과제명: {_safe_str(row.get('과제명'))}"]
    year = _safe_str(row.get("제출년도"))
    if year:
        parts.append(f"제출년도: {year}")
    kw = ", ".join(_str_list(row.get("키워드_리스트"))[:MAX_KEYWORDS])
    if kw:
        parts.append(f"키워드: {kw}")
    desc = clip_text(row.get("과제설명문"), FIELD_MAXLEN)
    if desc:
        parts.append(f"과제설명: {desc}")
    return "\n".join(parts)


# direction -> (source label, candidate label, which side carries the technology)
DIRECTION_SPEC = {
    "p2c": ("국가 R&D 과제", "기업", "source"),
    "c2p": ("기업", "국가 R&D 과제", "candidate"),
}


def build_score_messages(
    src_text: str,
    src_label: str,
    cands: Iterable[tuple[int, str]],
    cand_label: str,
    tech_side: str = "candidate",
    cand_maxlen: int = CAND_MAXLEN,
) -> list[dict[str, str]]:
    """system + user(source block, numbered candidate list, scoring guide).

    cands: [(id, text), ...] with 1-based ids, at most one batch (20) per prompt.
    """
    lines = [f"[평가 대상 {cand_label} 목록]"]
    lines += [f"{cid}. {clip_text(text, cand_maxlen)}" for cid, text in cands]
    tmpl = GUIDE_SOURCE_IS_TECH if tech_side == "source" else GUIDE_CANDIDATE_IS_TECH
    # the templates contain literal JSON braces, so str.format cannot be used
    guide = tmpl.replace("{src_label}", src_label).replace("{cand_label}", cand_label)
    user = f"[{src_label}]\n{src_text}\n\n" + "\n".join(lines) + f"\n\n{guide}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def strip_think(text: str) -> str:
    """Remove reasoning blocks some models emit before the answer."""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*</think>\s*", "", text)
    text = re.sub(r"<\|channel>thought.*?<channel\|>\s*", "", text, flags=re.DOTALL)
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def parse_scores(text: str) -> dict[int, tuple[int, str]]:
    """Robustly parse `{"results": [{"id", "score", "reason"}, ...]}` -> {id: (score, reason)}."""
    m = re.search(r'\{.*"results".*\}', text, re.DOTALL)
    blob = m.group(0) if m else text
    items: Sequence[Any]
    try:
        data = json.loads(blob)
        items = data.get("results", []) if isinstance(data, dict) else data
    except Exception:
        # fallback: pull out individual objects (handles truncated output)
        items = [
            {"id": int(mm.group(1)), "score": int(mm.group(2)), "reason": mm.group(3)}
            for mm in re.finditer(
                r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"score"\s*:\s*(\d+)\s*,\s*"reason"\s*:\s*"([^"]*)"',
                text,
            )
        ]
    out: dict[int, tuple[int, str]] = {}
    for it in items:
        try:
            i = int(it["id"])
            s = max(0, min(100, int(round(float(it["score"])))))
            out[i] = (s, str(it.get("reason", "")).strip()[:120])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
    return out
