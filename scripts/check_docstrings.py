"""Check that every exported function and class documents its parameters.

Ruff's pydocstyle rules check docstring form, but with the numpy convention they
cannot tell whether every argument is described (D417 is disabled by it). This
script does, for the public API that the documentation site renders: each name
in a package's ``__all__``, resolved to its definition, plus the public methods of
exported classes.

For every function or method, it reports
  - a missing docstring,
  - parameters missing from the ``Parameters`` section,
  - documented parameters that are not in the signature,
  - Sphinx roles (``:func:`x```), which do not render in the numpy style.

Classes may document constructor arguments in the class or ``__init__``
docstring; dataclass and pydantic fields go in an ``Attributes`` section.

Usage::

    uv run --isolated --only-group docs python scripts/check_docstrings.py [package ...]
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import griffe

sys.path.insert(0, str(Path(__file__).parent))
from generate_docs import discover_packages, export_targets  # noqa: E402

SPHINX_ROLE = re.compile(r":(?:func|class|meth|attr|mod|obj|data|exc):`")
IGNORED_PARAMS = {"self", "cls"}


def documented(docstring: griffe.Docstring | None, kind: griffe.DocstringSectionKind) -> set[str]:
    if docstring is None:
        return set()
    names: set[str] = set()
    for section in docstring.parse("numpy"):
        if section.kind is kind:
            for item in section.value:
                # "a, b : int" documents two parameters at once.
                names.update(n.strip() for n in item.name.split(","))
    return names


def check_function(func: griffe.Function, label: str, extra_doc: griffe.Docstring | None = None) -> list[str]:
    problems = []
    params = [
        p.name
        for p in func.parameters
        if p.name not in IGNORED_PARAMS and p.kind not in (griffe.ParameterKind.var_positional, griffe.ParameterKind.var_keyword)
    ]
    variadic = [p.name for p in func.parameters if p.kind in (griffe.ParameterKind.var_positional, griffe.ParameterKind.var_keyword)]
    if func.docstring is None and extra_doc is None:
        return [f"{label}: no docstring"]
    if func.docstring is not None and SPHINX_ROLE.search(func.docstring.value):
        problems.append(f"{label}: Sphinx role in docstring")
    names = documented(func.docstring, griffe.DocstringSectionKind.parameters)
    names |= documented(extra_doc, griffe.DocstringSectionKind.parameters)
    names |= documented(func.docstring, griffe.DocstringSectionKind.other_parameters)
    names = {n.lstrip("*") for n in names}
    missing = [p for p in params if p not in names]
    unknown = sorted(n for n in names if n not in params and n not in variadic)
    if missing:
        problems.append(f"{label}: undocumented parameters {missing}")
    if unknown:
        problems.append(f"{label}: documents parameters not in the signature {unknown}")
    return problems


def is_model(cls: griffe.Class) -> bool:
    """Dataclasses, NamedTuples and pydantic models: fields, not __init__ arguments."""
    bases = {str(b).rsplit(".", 1)[-1] for b in cls.bases}
    decorators = {str(d.value).split("(")[0].rsplit(".", 1)[-1] for d in cls.decorators}
    return bool(bases & {"BaseModel", "NamedTuple", "BaseModelTeamTomo", "TypedDict"}) or "dataclass" in decorators or any(
        "BaseModel" in b for b in bases
    )


def check_class(cls: griffe.Class, label: str) -> list[str]:
    problems = []
    if cls.docstring is None:
        problems.append(f"{label}: no docstring")
    elif SPHINX_ROLE.search(cls.docstring.value):
        problems.append(f"{label}: Sphinx role in docstring")
    if is_model(cls):
        fields = [
            name
            for name, member in cls.members.items()
            if member.is_attribute
            and not member.is_alias
            and member.annotation is not None
            and not name.startswith("_")
            and name != "model_config"
            and "ClassVar" not in str(member.annotation)
        ]
        names = documented(cls.docstring, griffe.DocstringSectionKind.attributes)
        if documented(cls.docstring, griffe.DocstringSectionKind.parameters):
            problems.append(f"{label}: model fields belong in an Attributes section, not Parameters")
        documented_inline = {name for name in fields if cls.members[name].docstring is not None}
        missing = [f for f in fields if f not in names and f not in documented_inline]
        if missing:
            problems.append(f"{label}: undocumented fields {missing}")
    elif "__init__" in cls.members and not cls.members["__init__"].is_alias:
        problems += check_function(cls.members["__init__"], f"{label}.__init__", extra_doc=cls.docstring)
    for name, member in cls.members.items():
        if name.startswith("_") or member.is_alias or not member.is_function:
            continue
        if {"property"} & member.labels:
            if member.docstring is None:
                problems.append(f"{label}.{name}: no docstring")
            continue
        problems += check_function(member, f"{label}.{name}")
    return problems


def main() -> None:
    logging.getLogger("griffe").setLevel(logging.ERROR)
    wanted = set(sys.argv[1:])
    packages = [p for p in discover_packages() if not wanted or p.name in wanted]
    search = [str(p.path / "src") for p in discover_packages()]
    loader = griffe.GriffeLoader(search_paths=search, docstring_parser=griffe.Parser.numpy)
    total = 0
    for pkg in packages:
        loader.load(pkg.module)
        problems: list[str] = []
        for path, kind in export_targets(pkg) or []:
            if kind not in ("function", "class"):
                continue
            try:
                obj = loader.modules_collection.get_member(path)
            except KeyError:
                problems.append(f"{path}: not found")
                continue
            label = path.rsplit(".", 1)[-1]
            problems += check_class(obj, label) if obj.is_class else check_function(obj, label)
        if problems:
            print(f"\n{pkg.name} ({len(problems)})")
            for problem in problems:
                print(f"  {problem}")
        total += len(problems)
    print(f"\n{total} problem(s)")
    sys.exit(1 if total else 0)


if __name__ == "__main__":
    main()
