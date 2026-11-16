"""Remove organisation names from description texts before embedding.

Project descriptions name the performing organisation ("이 과제는 A에서 수행했으며 …",
"A에서 수행하는 이 과제는 …") and company descriptions start with the company's own
name. An embedding of that text matches a project to its performing company by name
alone, which leaks held-out edges into every model that uses the vector.
"""
from __future__ import annotations

import re
from collections.abc import Iterable

ORG_PLACEHOLDER = "수행기관"
COMPANY_PLACEHOLDER = "이 회사"

_LEGAL_FORM = re.compile(
    r"\(주\)|\(유\)|\(재\)|\(사\)|\(합\)|\(사단\)|\(재단\)|㈜|주식회사|유한회사|유한책임회사|"
    r"합자회사|재단법인|사단법인|농업회사법인|영농조합법인"
)
_LEGAL_OPT = r"(?:\s*(?:" + _LEGAL_FORM.pattern + r")\s*)*"
# "<org>에서 수행" inside the opening sentence; the org is what precedes it
_PERFORMER = re.compile(r"^(?:이 (?:과제|연구|사업)는\s+)?(?P<org>[^.。\n]{1,80}?)에서\s*수행")


def core_name(name: object) -> str:
    """Name without legal-form markers and whitespace: '(주)가나 다' -> '가나다'."""
    return re.sub(r"\s+", "", _LEGAL_FORM.sub("", str(name or "")))


def _name_pattern(core: str) -> str:
    """Regex for a core name with optional whitespace inside and legal forms around it."""
    return _LEGAL_OPT + r"\s*".join(re.escape(ch) for ch in core) + _LEGAL_OPT


def _fix_particles(text: str, placeholder: str) -> str:
    """Make the particle after the placeholder agree with its final sound."""
    has_batchim = (ord(placeholder[-1]) - 0xAC00) % 28 != 0
    pairs = (("은", "는"), ("을", "를"), ("과", "와"))
    for with_b, without_b in pairs:
        wrong, right = (without_b, with_b) if has_batchim else (with_b, without_b)
        text = re.sub(re.escape(placeholder) + wrong + r"(?=[\s,])", placeholder + right, text)
    return text


def replace_names(text: str, names: Iterable[object], placeholder: str, min_len: int = 2) -> str:
    """Replace every occurrence of the given names (legal form and spacing ignored)."""
    cores = sorted({core_name(n) for n in names}, key=len, reverse=True)
    for core in cores:
        if len(core) >= min_len:
            text = re.sub(_name_pattern(core), placeholder, text)
    return _fix_particles(text, placeholder)


def performer_name(text: str) -> str | None:
    """Performing organisation named in the opening sentence of a project description."""
    m = _PERFORMER.match(text.strip())
    return m.group("org").strip() if m else None


def strip_project_names(text: str, linked_names: Iterable[object] = ()) -> str:
    """Project description with the performing organisation (and any linked company) removed."""
    names = list(linked_names)
    org = performer_name(text)
    if org:
        names.append(org)
    return replace_names(text, names, ORG_PLACEHOLDER)


def strip_company_name(text: str, name: object) -> str:
    """Company description with the company's own name replaced by a neutral phrase."""
    return replace_names(text, [name], COMPANY_PLACEHOLDER)
