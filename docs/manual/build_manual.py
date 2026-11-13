"""
Build the regc manual PDF from docs/manual/regc_manual.md.

1. Every ```mermaid block is written to figures/diagrams/<n>_<title>.mmd (the
   editable source, kept in the repository) and rendered to PNG with the
   Mermaid CLI.
2. A copy of the manual with each block replaced by its image (caption =
   the diagram's accTitle and accDescr) goes to _build/.
3. pandoc + XeLaTeX turn it into regc_manual.pdf.

The Markdown file itself keeps the Mermaid source, so it renders directly
on GitHub/GitLab and in editors with Mermaid support.

Usage:
    python docs/manual/build_manual.py [--mmdc "npx --prefix DIR mmdc"] [--chrome PATH]

Requires: pandoc, xelatex, the Libertinus fonts, and Mermaid CLI 12
(``npm install @mermaid-js/mermaid-cli@12.0.0``; set PUPPETEER_SKIP_DOWNLOAD=true
and pass --chrome to use an installed Chrome).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "regc_manual.md")
DIAGRAMS = os.path.join(HERE, "figures", "diagrams")
BUILD = os.path.join(HERE, "_build")
MERMAID_CONFIG = {"layout": "dagre", "theme": "default", "look": "classic",
                  "securityLevel": "strict"}

BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)


def slug(text: str) -> str:
    """A file-name-safe version of a diagram title."""
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:40] or "diagram"


def _width(png: str) -> str:
    """Print width of a rendered diagram: narrower for tall ones, so text stays near body size."""
    import struct
    with open(png, "rb") as f:
        w, h = struct.unpack(">II", f.read(24)[16:24])
    aspect = h / w
    if aspect > 1.8:
        return "32%"
    return "48%" if aspect > 1.2 else "70%" if aspect > 0.6 else "100%"


def render_diagrams(text: str, mmdc: str, chrome: str | None) -> str:
    """Write and render every Mermaid block; return the text with images instead."""
    os.makedirs(DIAGRAMS, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = os.path.join(tmp, "mermaid.json")
        with open(cfg, "w") as f:
            json.dump(MERMAID_CONFIG, f)
        pptr = os.path.join(tmp, "puppeteer.json")
        with open(pptr, "w") as f:
            json.dump({"args": ["--no-sandbox", "--headless=new"],
                       **({"executablePath": chrome} if chrome else {})}, f)
        counter = [0]

        def replace(m):
            body = m.group(1)
            counter[0] += 1
            title = re.search(r"accTitle:\s*(.+)", body)
            descr = re.search(r"accDescr:\s*(.+)", body)
            title = title.group(1).strip() if title else f"Diagram {counter[0]}"
            name = f"{counter[0]:02d}_{slug(title)}"
            mmd, png = (os.path.join(DIAGRAMS, name + ext) for ext in (".mmd", ".png"))
            with open(mmd, "w") as f:
                f.write(body)
            cmd = shlex.split(mmdc) + ["-i", mmd, "-o", png, "-c", cfg, "-p", pptr,
                                       "-s", "3", "-b", "white"]
            subprocess.run(cmd, check=True, capture_output=True)
            print("rendered", os.path.relpath(png, HERE))
            caption = f"**Diagram {counter[0]}.** {title}." + (
                f" {descr.group(1).strip()}" if descr else "")
            return f"![{caption}](figures/diagrams/{name}.png){{width={_width(png)}}}\n"

        return BLOCK.sub(replace, text)


def build_pdf(text: str) -> str:
    """pandoc + XeLaTeX; returns the PDF path."""
    os.makedirs(BUILD, exist_ok=True)
    # the PDF takes its title from the front matter: drop the Markdown H1 and contents list
    text = re.sub(r"^# regc user manual\n", "", text, count=1, flags=re.M)
    text = re.sub(r"\*\*Contents\*\*\n\n(?:\d+\. .*\n)+", "", text, count=1)
    md = os.path.join(BUILD, "regc_manual.md")
    with open(md, "w") as f:
        f.write(text)
    header = os.path.join(BUILD, "header.tex")
    with open(header, "w") as f:
        f.write(r"""\usepackage{caption}
\captionsetup{font=small,labelformat=empty,justification=justified,singlelinecheck=false}
\usepackage{float}\floatplacement{figure}{!htbp}
\renewcommand{\topfraction}{0.85}\renewcommand{\textfraction}{0.1}\renewcommand{\floatpagefraction}{0.75}
\renewcommand{\arraystretch}{1.25}
\usepackage{fvextra}\DefineVerbatimEnvironment{Highlighting}{Verbatim}{breaklines,commandchars=\\\{\}}
""")
    pdf = os.path.join(HERE, "regc_manual.pdf")
    cmd = ["pandoc", md, "-o", pdf, "--pdf-engine=xelatex", "--resource-path", HERE,
           "--toc", "--toc-depth=2", "--highlight-style=tango",
           "-V", "mainfont=Libertinus Serif", "-V", "sansfont=Libertinus Sans",
           "-V", "monofont=Libertinus Mono", "-V", "mathfont=Libertinus Math",
           "-V", "geometry:margin=2.2cm", "-V", "fontsize=10pt", "-V", "colorlinks=true",
           "-V", "linkcolor=blue!50!black", "-V", "urlcolor=blue!50!black",
           "-V", "toccolor=black", "-H", header]
    subprocess.run(cmd, check=True, cwd=HERE)
    print("wrote", os.path.relpath(pdf, HERE))
    return pdf


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--mmdc", default="npx mmdc", help="command running the Mermaid CLI")
    ap.add_argument("--chrome", default=None, help="Chrome/Chromium executable for Puppeteer")
    args = ap.parse_args()
    with open(SRC) as f:
        text = f.read()
    build_pdf(render_diagrams(text, args.mmdc, args.chrome))


if __name__ == "__main__":
    main()
