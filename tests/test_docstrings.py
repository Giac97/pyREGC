"""
Every module, class, function and method of the package has a NumPy-style
docstring: a summary; a Parameters section naming every argument; a Returns
(or Yields) section when it returns (yields) a value; a Raises section naming
the exceptions it raises. Properties need only a summary; functions nested
inside functions are exempt.

A light, dependency-free stand-in for ``numpydoc.validate``.
"""

import ast
import pathlib
import re
from typing import List, Set

import gcmc_re  # noqa: E402  -- check the package that is actually imported (repo or installed)

PACKAGE = pathlib.Path(gcmc_re.__file__).resolve().parent
SECTION = re.compile(r"^\s*([A-Z][A-Za-z ]+)\n\s*-{3,}\s*$", re.M)


def _sections(doc: str) -> Set[str]:
    """
    Section titles of a docstring.

    Parameters
    ----------
    doc : str
        The docstring.

    Returns
    -------
    set of str
        Titles such as ``'Parameters'``, ``'Returns'``.
    """
    return set(SECTION.findall(doc))


def _documented_params(doc: str) -> Set[str]:
    """
    Parameter names listed in the Parameters section.

    Parameters
    ----------
    doc : str
        The docstring.

    Returns
    -------
    set of str
        Names (several per line allowed, ``a, b : float``), without stars.
    """
    m = re.search(r"Parameters\n\s*-{3,}\n(.*?)(?:\n\s*\n\s*[A-Z][A-Za-z ]+\n\s*-{3,}|\Z)", doc, re.S)
    if not m:
        return set()
    names = set()
    for line in m.group(1).splitlines():
        if line.startswith(" " * 12):
            continue  # description lines
        mm = re.match(r"^\s*(\*{0,2}[A-Za-z_]\w*(?:\s*,\s*\*{0,2}[A-Za-z_]\w*)*)\s*(:|$)", line)
        if mm:
            names.update(n.strip().lstrip("*") for n in mm.group(1).split(","))
    return names


def _returns_value(fn: ast.AST) -> bool:
    """
    Whether a function body returns something other than None.

    Parameters
    ----------
    fn : ast.FunctionDef
        The function.

    Returns
    -------
    bool
        True if a ``return <value>`` occurs outside nested functions.
    """
    stack = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, ast.Return) and node.value is not None and not (
                isinstance(node.value, ast.Constant) and node.value.value is None):
            return True
        stack.extend(ast.iter_child_nodes(node))
    return False


def _raised(fn: ast.AST) -> Set[str]:
    """
    Exception class names raised directly in a function body.

    Parameters
    ----------
    fn : ast.FunctionDef
        The function.

    Returns
    -------
    set of str
        Names of explicitly raised exceptions (re-raises ignored).
    """
    out = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Raise) and node.exc is not None:
            exc = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(exc, ast.Name):
                out.add(exc.id)
            elif isinstance(exc, ast.Attribute):
                out.add(exc.attr)
    return out


def _check_function(path, fn, name: str, problems: List[str]) -> None:
    """
    Append the docstring problems of one function or method.

    Parameters
    ----------
    path : pathlib.Path
        Source file (for messages).
    fn : ast.FunctionDef
        The function.
    name : str
        Qualified name (for messages).
    problems : list of str
        Collected problems (appended to).
    """
    where = f"{path.name}:{fn.lineno} {name}"
    doc = ast.get_docstring(fn)
    if not doc:
        problems.append(f"{where}: no docstring")
        return
    if any((isinstance(d, ast.Name) and d.id == "property")
           or (isinstance(d, ast.Attribute) and d.attr == "setter") for d in fn.decorator_list):
        return
    args = [a.arg for a in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs
            if a.arg not in ("self", "cls")]
    args += [fn.args.vararg.arg] if fn.args.vararg else []
    args += [fn.args.kwarg.arg] if fn.args.kwarg else []
    secs = _sections(doc)
    if args:
        if "Parameters" not in secs:
            problems.append(f"{where}: no Parameters section for {args}")
        else:
            missing = [a for a in args if a not in _documented_params(doc)]
            if missing:
                problems.append(f"{where}: parameters not documented: {missing}")
    if fn.name != "__init__" and _returns_value(fn) and not secs & {"Returns", "Yields"}:
        problems.append(f"{where}: no Returns section")
    raised = _raised(fn) - {"SystemExit"}
    if raised:
        raises_text = doc[doc.find("Raises"):] if "Raises" in secs else ""
        missing = sorted(e for e in raised if e not in raises_text)
        if missing:
            problems.append(f"{where}: Raises section does not name {missing}")


def collect_problems(package_dir: pathlib.Path = PACKAGE) -> List[str]:
    """
    All docstring problems in a package directory.

    Parameters
    ----------
    package_dir : pathlib.Path, optional
        Directory of ``*.py`` files. Default: the gcmc_re package.

    Returns
    -------
    list of str
        One ``file:line name: problem`` line per problem.
    """
    problems: List[str] = []
    for path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(path.read_text())
        if not ast.get_docstring(tree):
            problems.append(f"{path.name}: no module docstring")

        def visit(node, prefix):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    if not ast.get_docstring(child):
                        problems.append(f"{path.name}:{child.lineno} class {prefix}{child.name}: "
                                        f"no docstring")
                    visit(child, f"{prefix}{child.name}.")
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    _check_function(path, child, prefix + child.name, problems)

        visit(tree, "")
    return problems


def test_package_docstrings_are_numpy_style():
    assert len(list(PACKAGE.glob("*.py"))) > 10, f"package not found at {PACKAGE}"
    problems = collect_problems()
    assert not problems, "docstring problems:\n" + "\n".join(problems)


def test_checker_catches_problems(tmp_path):
    (tmp_path / "bad.py").write_text('''"""Module."""


def f(a, b):
    """
    Summary.

    Parameters
    ----------
    a : int
        A.
    """
    if a:
        raise KeyError(a)
    return a + b


class C:
    def g(self):
        return 1
''')
    found = "\n".join(collect_problems(tmp_path))
    assert "['b']" in found and "no Returns" in found and "KeyError" in found
    assert "class C" in found and "C.g: no docstring" in found
