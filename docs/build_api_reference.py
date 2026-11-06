"""
Generate docs/API_REFERENCE.md from the package's NumPy-style docstrings.

Run from the repository root after changing a public signature or docstring:

    python docs/build_api_reference.py

Every public name in ``regc.__all__`` is documented, grouped by module, plus
the public functions of ``regc.runner``. Docstring sections (Parameters,
Returns, Raises, ...) become Markdown lists; everything else is kept as text.
"""

from __future__ import annotations

import inspect
import os
import re
import sys
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import regc  # noqa: E402
from regc import runner  # noqa: E402

MODULE_ORDER = [
    ("calculators", "Calculators"),
    ("lammps_backend", "LAMMPS backend"),
    ("systems", "Systems"),
    ("structures", "Structures and slab geometry"),
    ("state", "Thermodynamic state"),
    ("replica", "Replicas"),
    ("engine", "The replica-exchange engine"),
    ("driver", "Checkpointed runs"),
    ("io", "Sample I/O"),
    ("units", "Units"),
    ("pressure", "Reduced-unit pressure conventions"),
    ("reservoir", "Gas reservoirs"),
    ("analysis", "MBAR analysis"),
    ("plotting", "Plotting"),
]
RUNNER_FUNCS = ["load_config", "validate_config", "setup_from_config", "run_from_config",
                "analyze_from_config", "check_config", "main"]
LIST_SECTIONS = {"Parameters", "Other Parameters", "Returns", "Yields", "Raises", "Attributes",
                 "Warns"}


def rst_to_md(text: str) -> str:
    """
    Convert the RST inline markup used in the docstrings to Markdown.

    Parameters
    ----------
    text : str
        Docstring text.

    Returns
    -------
    str
        The text with ``double backticks`` turned into single ones.
    """
    return re.sub(r"``([^`]+)``", r"`\1`", text)


def split_sections(doc: str):
    """
    Split a NumPy docstring into its free text and named sections.

    Parameters
    ----------
    doc : str
        The cleaned docstring (`inspect.getdoc`).

    Returns
    -------
    tuple
        ``(intro_lines, [(section_name, lines), ...])``.
    """
    lines = doc.splitlines()
    intro: List[str] = []
    sections = []
    current = None
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and set(nxt.strip()) == {"-"} and len(nxt.strip()) >= 3:
            current = (line.strip(), [])
            sections.append(current)
            i += 2
            continue
        (current[1] if current else intro).append(line)
        i += 1
    return intro, sections


def render_list_section(name: str, lines: List[str]) -> str:
    """
    Render a Parameters/Returns/Raises-style section as a Markdown list.

    Parameters
    ----------
    name : str
        Section name.
    lines : list of str
        The section's lines.

    Returns
    -------
    str
        Markdown.
    """
    items = []
    for line in lines:
        if not line.strip():
            if items:
                items[-1][1].append("")
            continue
        if not line.startswith(" "):
            head = line.strip()
            if " : " in head:
                pname, ptype = head.split(" : ", 1)
                label = f"`{pname.strip()}` (*{rst_to_md(ptype.strip())}*)"
            elif name in ("Returns", "Yields", "Raises", "Warns"):
                label = f"*{rst_to_md(head)}*"
            else:
                label = f"`{head}`"
            items.append((label, []))
        elif items:
            items[-1][1].append(line.strip())
    out = [f"**{name}**", ""]
    for label, desc in items:
        text = " ".join(d for d in desc if d).strip()
        out.append(f"- {label}" + (f": {rst_to_md(text)}" if text else ""))
    return "\n".join(out)


def render_doc(doc: str) -> str:
    """
    Render a whole docstring.

    Parameters
    ----------
    doc : str
        The docstring, or an empty string.

    Returns
    -------
    str
        Markdown.
    """
    if not doc:
        return "_No docstring._"
    intro, sections = split_sections(doc)
    parts = [rst_to_md("\n".join(intro).strip())]
    for name, lines in sections:
        if name in LIST_SECTIONS:
            parts.append(render_list_section(name, lines))
        else:
            parts.append(f"**{name}**\n\n" + rst_to_md("\n".join(lines).strip()))
    return "\n\n".join(p for p in parts if p)


def signature(obj, name: str, drop_self: bool = False) -> str:
    """
    A code block with the signature of a callable.

    Parameters
    ----------
    obj : callable
        The function, method or class.
    name : str
        Name to show.
    drop_self : bool, optional
        Remove a leading ``self`` parameter. Default False.

    Returns
    -------
    str
        Markdown code block.
    """
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return f"```python\n{name}(...)\n```"
    params = list(sig.parameters.values())
    if drop_self and params and params[0].name == "self":
        params = params[1:]
    sig = sig.replace(parameters=params)
    text = re.sub(r" at 0x[0-9a-f]+>", ">", f"{name}{sig}")
    text = re.sub(r"""(:\s*|->\s*)"?'([^'"]+)'"?""", r"\1\2", text)   # unquote annotations
    return f"```python\n{text}\n```"


def document_class(cls, qual: str) -> str:
    """
    Document a class: constructor, then public methods and properties.

    Parameters
    ----------
    cls : type
        The class.
    qual : str
        Name to show.

    Returns
    -------
    str
        Markdown.
    """
    out = [f"### `{qual}`", "", signature(cls.__init__, qual, drop_self=True), "",
           render_doc(inspect.getdoc(cls) or "")]
    init_doc = cls.__dict__.get("__init__").__doc__ if "__init__" in cls.__dict__ else None
    if init_doc:
        out += ["", "**Constructor**", "", render_doc(inspect.cleandoc(init_doc))]
    members = []
    for mname, member in cls.__dict__.items():
        if mname.startswith("_"):
            continue
        if isinstance(member, property):
            members.append(f"#### `{qual}.{mname}` (property)\n\n"
                           + render_doc(inspect.getdoc(member) or ""))
        elif isinstance(member, (staticmethod, classmethod)) or inspect.isfunction(member):
            fn = getattr(cls, mname)
            drop = inspect.isfunction(member)
            kind = " (classmethod)" if isinstance(member, classmethod) else (
                " (staticmethod)" if isinstance(member, staticmethod) else "")
            members.append(f"#### `{qual}.{mname}`{kind}\n\n"
                           + signature(fn, f"{qual}.{mname}", drop_self=drop) + "\n\n"
                           + render_doc(inspect.getdoc(fn) or ""))
    return "\n".join(out) + ("\n\n" + "\n\n".join(members) if members else "")


def document(obj, name: str) -> str:
    """
    Document one public name.

    Parameters
    ----------
    obj : object
        Class, function or constant.
    name : str
        Its public name.

    Returns
    -------
    str
        Markdown.
    """
    if inspect.isclass(obj) and obj.__name__ != name:
        return (f"### `{name}`\n\nAlias of [`{obj.__name__}`](#{obj.__name__.lower()}), "
                "kept for older scripts.")
    if inspect.isclass(obj):
        return document_class(obj, name)
    if callable(obj):
        return (f"### `{name}`\n\n" + signature(obj, name) + "\n\n"
                + render_doc(inspect.getdoc(obj) or ""))
    return f"### `{name}`\n\n```python\n{name} = {obj!r}\n```"


def main() -> None:
    """
    Write docs/API_REFERENCE.md.

    Returns
    -------
    None
    """
    by_module = {}
    for name in regc.__all__:
        obj = getattr(regc, name)
        mod = getattr(obj, "__module__", None) or ""
        if not mod.startswith("regc."):
            # constants: find the defining module by attribute lookup
            mod = next((f"regc.{m}" for m, _ in MODULE_ORDER
                        if name in vars(__import__(f"regc.{m}", fromlist=["x"]))), "regc")
        by_module.setdefault(mod.split(".", 1)[-1], []).append((name, obj))

    toc, body = [], []
    for mod, title in MODULE_ORDER:
        entries = by_module.pop(mod, [])
        if not entries:
            continue
        anchor = re.sub(r"[^a-z0-9 _-]", "", f"{title} regc.{mod}".lower()).replace(" ", "-")
        toc.append(f"- [{title} (`regc.{mod}`)](#{anchor}): "
                   + ", ".join(f"`{n}`" for n, _ in entries))
        body.append(f"## {title} (`regc.{mod}`)\n\n"
                    + "\n\n---\n\n".join(document(o, n) for n, o in entries))
    for mod, entries in by_module.items():  # anything not in MODULE_ORDER
        body.append(f"## `regc.{mod}`\n\n" + "\n\n---\n\n".join(document(o, n) for n, o in entries))

    toc.append("- [Config runner (`regc.runner`)](#config-runner-regcrunner): "
               + ", ".join(f"`{n}`" for n in RUNNER_FUNCS))
    body.append("## Config runner (`regc.runner`)\n\n"
                "Not exported at the package top level: `from regc.runner import ...`. "
                "The config file format is in MANUAL.md, Part II.\n\n"
                + "\n\n---\n\n".join(document(getattr(runner, n), n) for n in RUNNER_FUNCS))

    header = ("# regc API reference\n\n"
              "Generated from the docstrings of regc by "
              "`docs/build_api_reference.py`; do not edit by hand. For how to use these "
              "pieces together, see [MANUAL.md](MANUAL.md).\n\n"
              "Everything below is importable as `from regc import <name>`, except the "
              "runner functions at the end.\n\n## Contents\n\n" + "\n".join(toc))
    with open(os.path.join(HERE, "API_REFERENCE.md"), "w") as f:
        f.write(header + "\n\n" + "\n\n".join(body) + "\n")
    print(f"wrote {os.path.join(HERE, 'API_REFERENCE.md')}")


if __name__ == "__main__":
    main()
