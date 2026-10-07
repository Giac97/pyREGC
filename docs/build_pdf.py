"""
Convert docs/REGC_algorithm.md to PDF via XeLaTeX (no pandoc needed).

    python docs/build_pdf.py [input.md] [output.pdf]

Handles exactly the Markdown subset the doc uses: # / ## / ### headings,
paragraphs, `*` bullet lists (nested by 2-space indent), pipe tables, fenced
code blocks, > block quotes, --- rules, inline `code` / **bold** / *italic* /
$math$, and $$ display math $$ whose trailing "\\qquad (n)" becomes \\tag{n}.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage{fontspec}
\setmonofont{FreeMono}[Scale=0.92]
\usepackage{booktabs,tabularx,array,enumitem,fancyvrb,xcolor}
\usepackage[colorlinks=true,linkcolor=blue!50!black,urlcolor=blue!50!black]{hyperref}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}
\setlist[itemize]{itemsep=2pt,topsep=4pt}
\setlength{\parskip}{4pt}\setlength{\parindent}{0pt}
\fvset{fontsize=\small,frame=single,framesep=6pt,rulecolor=\color{black!30}}
\begin{document}
"""

TEXT_UNICODE = {
    "→": r"$\to$", "←": r"$\leftarrow$", "≤": r"$\le$", "≥": r"$\ge$", "∈": r"$\in$",
    "ρ": r"$\rho$", "μ": r"$\mu$", "…": r"\dots{}", "₂": r"$_2$", "Å": r"\AA{}",
    "–": "--", "—": "---", "×": r"$\times$",
}


def escape_text(s: str) -> str:
    out = []
    for ch in s:
        if ch in "&%#_{}":
            out.append("\\" + ch)
        elif ch == "~":
            out.append(r"\textasciitilde{}")
        elif ch == "^":
            out.append(r"\textasciicircum{}")
        elif ch == "\\":
            out.append(r"\textbackslash{}")
        else:
            out.append(TEXT_UNICODE.get(ch, ch))
    return "".join(out)


def escape_code(s: str) -> str:
    body = escape_text(s).replace("--", "-{}-")
    body = body.replace("\\_", "\\_\\allowbreak{}").replace(".", ".\\allowbreak{}")
    return r"\texttt{" + body + "}"


def emphasis(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    return re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\\emph{\1}", s)


def inline(s: str) -> str:
    """Escape a line of text, keeping `code` and $math$ spans intact."""
    tokens = re.split(r"(`[^`]+`|\$[^$]+\$)", s)
    placeholders, text = [], []
    for tok in tokens:
        if tok.startswith("`") and tok.endswith("`") and len(tok) > 1:
            placeholders.append(escape_code(tok[1:-1]))
        elif tok.startswith("$") and tok.endswith("$") and len(tok) > 1:
            placeholders.append(tok)
        else:
            text.append(escape_text(tok))
            continue
        text.append(f"\x00{len(placeholders) - 1}\x00")
    joined = emphasis("".join(text))
    joined = re.sub(r'"([^"]*)"', r"``\1''", joined)
    return re.sub(r"\x00(\d+)\x00", lambda m: placeholders[int(m.group(1))], joined)


def table(rows: list) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], cells[2:]
    spec = "".join("l" if h == "#" else "Y" for h in header)
    lines = [r"\begin{center}\small", rf"\begin{{tabularx}}{{\linewidth}}{{{spec}}}", r"\toprule",
             " & ".join(rf"\textbf{{{inline(h)}}}" for h in header) + r" \\", r"\midrule"]
    lines += [" & ".join(inline(c) for c in r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabularx}", r"\end{center}"]
    return "\n".join(lines)


def bullet_list(lines: list) -> str:
    items = []  # (level, text)
    for line in lines:
        m = re.match(r"^(\s*)\* (.*)$", line)
        if m:
            items.append([len(m.group(1)) // 2, m.group(2)])
        else:
            items[-1][1] += " " + line.strip()
    out, level = [], -1
    for lvl, text in items:
        while level < lvl:
            out.append(r"\begin{itemize}")
            level += 1
        while level > lvl:
            out.append(r"\end{itemize}")
            level -= 1
        out.append(r"\item " + inline(text))
    out += [r"\end{itemize}"] * (level + 1)
    return "\n".join(out)


def display_math(body: str) -> str:
    m = re.search(r"\\qquad\s*\((\d+)\)\s*$", body.strip())
    if m:
        body = body.strip()[: m.start()].rstrip()
        return "\\begin{equation}\n" + body + f"\n\\tag{{{m.group(1)}}}\n\\end{{equation}}"
    return "\\[\n" + body.strip() + "\n\\]"


def convert(md: str) -> str:
    lines = md.splitlines()
    out, i = [PREAMBLE], 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            j = i + 1
            while not lines[j].startswith("```"):
                j += 1
            out.append("\\begin{Verbatim}\n" + "\n".join(lines[i + 1:j]) + "\n\\end{Verbatim}")
            i = j + 1
        elif line.strip() == "$$":
            j = i + 1
            while lines[j].strip() != "$$":
                j += 1
            out.append(display_math("\n".join(lines[i + 1:j])))
            i = j + 1
        elif line.startswith("# "):
            out.append(r"{\LARGE\bfseries " + inline(line[2:]) + r"\par}\medskip")
            i += 1
        elif line.startswith("## "):
            out.append(r"\section*{" + inline(line[3:]) + "}")
            i += 1
        elif line.startswith("### "):
            out.append(r"\subsection*{" + inline(line[4:]) + "}")
            i += 1
        elif line.strip() == "---":
            out.append(r"\medskip\noindent\rule{\linewidth}{0.3pt}")
            i += 1
        elif line.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            out.append(table(lines[i:j]))
            i = j
        elif re.match(r"^\s*\* ", line):
            j = i
            while j < len(lines) and lines[j].strip() and (re.match(r"^\s*\* ", lines[j])
                                                          or lines[j].startswith(" ")):
                j += 1
            out.append(bullet_list(lines[i:j]))
            i = j
        elif line.startswith(">"):
            j = i
            while j < len(lines) and lines[j].startswith(">"):
                j += 1
            text = " ".join(l.lstrip(">").strip() for l in lines[i:j])
            out.append(r"\begin{quote}" + inline(text) + r"\end{quote}")
            i = j
        else:
            j = i
            while (j < len(lines) and lines[j].strip() and lines[j].strip() != "$$"
                   and not lines[j].startswith(("#", "|", "```", ">"))
                   and not re.match(r"^\s*\* ", lines[j])):
                j += 1
            out.append(inline(" ".join(l.strip() for l in lines[i:j])) + "\n")
            i = j
    out.append(r"\end{document}")
    return "\n\n".join(out)


def main() -> None:
    here = pathlib.Path(__file__).resolve().parent
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else here / "REGC_algorithm.md"
    dst = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    tex = convert(src.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        tex_path = pathlib.Path(tmp) / "doc.tex"
        tex_path.write_text(tex, encoding="utf-8")
        for _ in range(2):  # second pass settles hyperref
            res = subprocess.run(["xelatex", "-interaction=nonstopmode", "-halt-on-error",
                                  tex_path.name], cwd=tmp, capture_output=True, text=True)
            if res.returncode != 0:
                sys.exit(res.stdout[-3000:])
        shutil.copy(pathlib.Path(tmp) / "doc.pdf", dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
