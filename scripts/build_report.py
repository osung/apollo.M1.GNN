"""Render the experiment report (Markdown) to .docx.

Supports the Markdown subset used in reports/: #/##/### headings, paragraphs,
bullet and numbered lists, pipe tables, fenced code, **bold** and `code`.
Several input files are concatenated in order (e.g. the report body followed
by the expected-results appendix).

    python scripts/build_report.py reports/a.md reports/b.md -o reports/out.docx

Needs python-docx (not a runtime dependency of the project).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor  # noqa: I001

INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`)")
BODY_PT, CODE_PT, TABLE_PT = 10.5, 9.0, 9.0


def set_font(run, name: str, size: float | None = None) -> None:
    run.font.name = name
    r_fonts = run._element.get_or_add_rPr().get_or_add_rFonts()
    r_fonts.set(qn("w:eastAsia"), name)
    if size is not None:
        run.font.size = Pt(size)


def add_inline(par, text: str, font: str, mono: str, size: float) -> None:
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = par.add_run(part[2:-2])
            run.bold = True
            set_font(run, font, size)
        elif part.startswith("`") and part.endswith("`"):
            run = par.add_run(part[1:-1])
            set_font(run, mono, size - 1)
        else:
            set_font(par.add_run(part), font, size)


def shade(cell, fill: str) -> None:
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    cell._element.get_or_add_tcPr().append(shd)


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def render(doc: Document, lines: list[str], font: str, mono: str) -> None:
    i = 0
    while i < len(lines):
        line = lines[i].rstrip("\n")
        stripped = line.strip()
        if not stripped:
            i += 1
        elif stripped.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                par = doc.add_paragraph()
                par.paragraph_format.space_after = Pt(0)
                set_font(par.add_run(lines[i].rstrip("\n")), mono, CODE_PT)
                i += 1
            i += 1
        elif stripped.startswith("#"):
            level = len(stripped) - len(stripped.lstrip("#"))
            text = stripped[level:].strip().replace("`", "")
            if level == 1:
                par = doc.add_paragraph()
                par.alignment = WD_ALIGN_PARAGRAPH.CENTER
                par.paragraph_format.space_after = Pt(12)
                run = par.add_run(text)
                run.bold = True
                set_font(run, font, 18)
            else:
                head = doc.add_heading(level=level - 1)
                run = head.add_run(text)
                run.font.color.rgb = RGBColor(0x1F, 0x2D, 0x3D)
                set_font(run, font, {2: 15, 3: 12.5}.get(level, 11))
            i += 1
        elif stripped.startswith("|") and i + 1 < len(lines) and re.match(
            r"^\s*\|[\s:|-]+\|\s*$", lines[i + 1]
        ):
            header = split_row(stripped)
            rows = []
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i]))
                i += 1
            table = doc.add_table(rows=1, cols=len(header))
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            for cell, text in zip(table.rows[0].cells, header, strict=True):
                shade(cell, "E8EDF3")
                par = cell.paragraphs[0]
                run = par.add_run(text.replace("**", ""))
                run.bold = True
                set_font(run, font, TABLE_PT)
            for row in rows:
                cells = table.add_row().cells
                for cell, text in zip(cells, row + [""] * (len(header) - len(row)), strict=False):
                    add_inline(cell.paragraphs[0], text, font, mono, TABLE_PT)
            doc.add_paragraph().paragraph_format.space_after = Pt(0)
        elif re.match(r"^\s*[-*] ", line):
            indent = len(line) - len(line.lstrip())
            par = doc.add_paragraph(style="List Bullet 2" if indent >= 2 else "List Bullet")
            add_inline(par, re.sub(r"^\s*[-*] ", "", line), font, mono, BODY_PT)
            i += 1
        elif re.match(r"^\s*\d+\. ", line):
            # literal numbers: Word's "List Number" style keeps counting across lists
            par = doc.add_paragraph()
            par.paragraph_format.left_indent = Pt(22)
            par.paragraph_format.first_line_indent = Pt(-14)
            add_inline(par, stripped, font, mono, BODY_PT)
            i += 1
        else:
            par = doc.add_paragraph()
            add_inline(par, stripped, font, mono, BODY_PT)
            i += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="Markdown files, concatenated in order")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--font", default="Noto Sans CJK KR")
    parser.add_argument("--mono-font", default="Noto Sans Mono CJK KR")
    args = parser.parse_args()

    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = args.font
    normal.font.size = Pt(BODY_PT)
    normal.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), args.font)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25

    for n, path in enumerate(args.inputs):
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if n > 0:
            doc.add_page_break()
        render(doc, lines, args.font, args.mono_font)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    doc.save(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
