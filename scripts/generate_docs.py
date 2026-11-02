"""Generate the per-package pages of the documentation site.

Zensical builds the site from ``docs/``. Most of what lives there is written by
hand, but every package contributes pages that are generated from files the
package already owns, so nothing is written twice:

    packages/<category>/<package>/
        README.md            -> docs/<category>/<package>/index.md
        docs/*.md            -> docs/<category>/<package>/*.md      (optional)
        docs/api.md          -> replaces the default API reference  (optional)
        examples/*.ipynb     -> docs/<category>/<package>/examples/*.md
        examples/*.py        -> (rendered as code, docstring as intro)
        docs/examples/*      -> examples too; Markdown is copied as is

The default API reference renders the package's top-level ``__all__``.

The I/O packages live in their own repositories. They are listed in
``scripts/io_packages.toml``, cloned into ``external/`` (git-ignored) when missing,
and generated under ``docs/io/`` by the same rules. ``--update`` pulls the latest
default branch of each clone.

Contributor documentation is generated from ``CONTRIBUTING.md`` and ``notes/``.
``docs/examples/index.md`` is a hand-written placeholder; the top-level
``examples/`` are not rendered yet.
Notebooks are rendered from their stored outputs; they are never executed.

Generated directories are listed in ``GENERATED_DIRS`` and git-ignored. The
navigation in ``zensical.toml`` is written by hand; this script fails if a
generated page is missing from it, and ``--print-nav`` prints the entries to
paste in.

Usage::

    uv run --isolated --only-group docs python scripts/generate_docs.py
    uv run --isolated --only-group docs zensical serve
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
REPO_URL = "https://github.com/teamtomo/teamtomo"
EXTERNAL = ROOT / "external"
IO_PACKAGES = ROOT / "scripts" / "io_packages.toml"
CATEGORIES = {
    "primitives": "Primitives",
    "algorithms": "Algorithms",
    "io": "I/O",
    "utils": "Utilities",
}
GENERATED_DIRS = [*CATEGORIES, "developer"]
NOTES_SKIP = re.compile(r"\.local\.md$")
WARNINGS: list[str] = []
# Local checkout root -> GitHub URL prefix (with a {kind} of blob or tree). The
# deepest matching root wins, so clones under external/ map to their own repo.
SOURCE_ROOTS: dict[Path, str] = {ROOT: f"{REPO_URL}/{{kind}}/main"}
# Package name -> checkout, for links to the old per-package sites.
LEGACY_SITES: dict[str, Path] = {}


def warn(message: str) -> None:
    WARNINGS.append(message)
    print(f"warning: {message}", file=sys.stderr)


# Markdown links and images whose target is a relative path.
MD_LINK = re.compile(
    r"(!?\[[^\]]*\]\()(?!https?://|mailto:|#|/)([^)\s]+)(\s+\"[^\"]*\")?\)"
)
HTML_SRC = re.compile(
    r"""(<(?:img|source)\b[^>]*?\bsrc=["'])(?!https?://|/)([^"']+)(["'])"""
)
# Links to the per-package sites that these docs replace, e.g. teamtomo.org/starfile/.
LEGACY_LINK = re.compile(
    r"(\]\()https?://teamtomo\.org/([a-z0-9_-]+)/?([^)#\s]*)(#[^)\s]*)?\)"
)
BADGE_LINE = re.compile(r"^\s*(\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)\s*)+$")


@dataclass
class Package:
    name: str
    category: str
    path: Path
    module: str
    description: str
    src: Path | None = None  # directory holding the module; default <path>/src
    group: str = ""
    out: Path = field(init=False)
    pages: list[tuple[str, str]] = field(
        default_factory=list
    )  # (title, docs-relative path)
    examples: list[tuple[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.out = DOCS / self.category / self.name
        if self.src is None:
            self.src = self.path / "src"


def github_url(path: Path) -> str | None:
    """GitHub URL of a file or directory in the monorepo or an external clone."""
    for root in sorted(SOURCE_ROOTS, key=lambda r: len(r.parts), reverse=True):
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        kind = "tree" if path.is_dir() else "blob"
        return f"{SOURCE_ROOTS[root].format(kind=kind)}/{rel.as_posix()}"
    return None


def source_label(path: Path) -> str:
    """Path of a file within its own repository, prefixed for external ones."""
    for root in sorted(SOURCE_ROOTS, key=lambda r: len(r.parts), reverse=True):
        if path.is_relative_to(root):
            rel = path.relative_to(root).as_posix()
            return rel if root == ROOT else f"{root.name}/{rel}"
    return path.name


def discover_packages() -> list[Package]:
    packages = []
    for category in CATEGORIES:
        for pyproject in sorted(
            (ROOT / "packages" / category).glob("*/pyproject.toml")
        ):
            meta = tomllib.loads(pyproject.read_text())["project"]
            inits = sorted((pyproject.parent / "src").glob("*/__init__.py"))
            if not inits:
                sys.exit(f"{pyproject.parent}: no src/<module>/__init__.py")
            packages.append(
                Package(
                    name=meta["name"],
                    category=category,
                    path=pyproject.parent,
                    module=inits[0].parent.name,
                    description=meta.get("description", "").strip(),
                )
            )
    return packages


def git(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )
    if result.returncode:
        sys.exit(f"git {' '.join(args)} failed:\n{result.stderr}")
    return result.stdout.strip()


def discover_io_packages(update: bool) -> list[Package]:
    """Clone (or with ``update``, refresh) the I/O repositories and describe them."""
    packages = []
    for entry in tomllib.loads(IO_PACKAGES.read_text())["package"]:
        name, repo = entry["name"], entry["repo"]
        checkout = EXTERNAL / name
        if not checkout.exists():
            print(f"cloning {repo}", file=sys.stderr)
            EXTERNAL.mkdir(exist_ok=True)
            git("clone", "--quiet", "--depth", "1", repo, str(checkout))
        elif update:
            git("fetch", "--quiet", "--depth", "1", "origin", "HEAD", cwd=checkout)
            git("reset", "--quiet", "--hard", "FETCH_HEAD", cwd=checkout)
        branch = git("rev-parse", "--abbrev-ref", "HEAD", cwd=checkout)
        SOURCE_ROOTS[checkout] = f"{repo}/{{kind}}/{branch}"
        LEGACY_SITES[name] = checkout
        # src/ layout, or the module at the top of the repository.
        src = checkout / "src" if (checkout / "src").is_dir() else checkout
        module = name.replace("-", "_")
        if not (src / module / "__init__.py").exists():
            sys.exit(f"{checkout}: no {module}/__init__.py under {src}")
        packages.append(
            Package(
                name=name,
                category="io",
                path=checkout,
                module=module,
                description=entry["description"],
                src=src,
                group=entry.get("group", ""),
            )
        )
    return packages


FENCE = re.compile(r"^[ \t]*(```|~~~).*?^[ \t]*\1[^\n]*$", re.MULTILINE | re.DOTALL)


def outside_code(text: str, fn) -> str:
    """Apply ``fn`` to the parts of ``text`` that are not fenced code blocks."""
    out, last = [], 0
    for m in FENCE.finditer(text):
        out += [fn(text[last : m.start()]), m.group(0)]
        last = m.end()
    return "".join(out + [fn(text[last:])])


LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+\S")


def separate_lists(text: str) -> str:
    """Put a blank line before a list that directly follows a paragraph.

    GitHub renders such lists; Python-Markdown folds them into the paragraph.
    """

    def fix(chunk: str) -> str:
        out: list[str] = []
        for line in chunk.split("\n"):
            prev = out[-1] if out else ""
            if (
                LIST_ITEM.match(line)
                and prev.strip()
                and not LIST_ITEM.match(prev)
                and not prev.startswith((" ", "\t", "#", "|", ">", "<"))
            ):
                out.append("")
            out.append(line)
        return "\n".join(out)

    return outside_code(text, fix)


def title_of(markdown: str, fallback: str) -> str:
    prose = FENCE.sub("", markdown)
    match = re.search(r"^#\s+(.+?)\s*$", prose, re.MULTILINE)
    return match.group(1).strip("` ") if match else fallback


def rewrite_links(
    text: str, src_dir: Path, out_dir: Path, page_map: dict[Path, Path], assets: list
) -> str:
    """Point relative links in ``text`` (from ``src_dir``) at the right target.

    Links to files that become pages are rewritten to those pages, images are
    copied next to the generated page, and anything else goes to GitHub.
    """

    def resolve(target: str) -> str:
        path_part, _, anchor = target.partition("#")
        source = (src_dir / path_part).resolve()
        if source in page_map:
            rel = Path(_relpath(page_map[source], out_dir))
            return f"{rel.as_posix()}{'#' + anchor if anchor else ''}"
        if source.is_file() and source.suffix.lower() in {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
        }:
            dest = out_dir / path_part
            assets.append((source, dest))
            return path_part
        if not source.exists():
            warn(f"{src_dir.relative_to(ROOT)}: broken link {target}")
        url = github_url(source)
        if url is None:
            return target
        return f"{url}{'#' + anchor if anchor else ''}"

    def legacy(match: re.Match) -> str:
        """Point a link to an old per-package site at the page that replaced it."""
        name, page, anchor = match.group(2), match.group(3).strip("/"), match.group(4)
        checkout = LEGACY_SITES.get(name)
        if checkout is None:
            return match.group(0)
        candidates = (
            [checkout / "README.md"]
            if not page
            else [
                checkout / "docs" / f"{page}.md",
                checkout / "docs" / page / "index.md",
            ]
        )
        # Emit the source file; the relative-link pass maps it to its page.
        for candidate in candidates:
            if candidate.resolve() in page_map:
                rel = _relpath(candidate, src_dir)
                return f"{match.group(1)}{rel}{anchor or ''})"
        return match.group(0)

    def rewrite(chunk: str) -> str:
        chunk = LEGACY_LINK.sub(legacy, chunk)
        chunk = MD_LINK.sub(
            lambda m: f"{m.group(1)}{resolve(m.group(2))}{m.group(3) or ''})", chunk
        )
        return HTML_SRC.sub(
            lambda m: f"{m.group(1)}{resolve(m.group(2))}{m.group(3)}", chunk
        )

    return outside_code(text, rewrite)


def _relpath(target: Path, start: Path) -> str:
    import os

    return os.path.relpath(target, start)


def strip_badges(text: str) -> str:
    """Drop lines made only of badges; PyPI shows them, the docs don't need them."""
    return (
        "\n".join(line for line in text.splitlines() if not BADGE_LINE.match(line))
        + "\n"
    )


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def render_notebook(notebook: Path, out_md: Path) -> str:
    """Convert a notebook to Markdown from its stored outputs; return its title."""
    import nbformat
    from nbconvert import MarkdownExporter

    nb = nbformat.read(notebook, as_version=4)
    resources = {"output_files_dir": f"{out_md.stem}_files"}
    body, resources = MarkdownExporter().from_notebook_node(nb, resources=resources)
    body = separate_lists(body)
    for name, data in resources.get("outputs", {}).items():
        dest = out_md.parent / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    rel = source_label(notebook)
    title = title_of(body, notebook.stem.replace("_", " ").capitalize())
    banner = (
        f'!!! note "Notebook"\n'
        f"    Rendered from [`{rel}`]({github_url(notebook)}) with its stored outputs. "
        f"Download it to run it yourself.\n\n"
    )
    first_cell = nb.cells[0] if nb.cells else None
    if (
        first_cell is not None
        and first_cell.cell_type == "markdown"
        and first_cell.source.lstrip().startswith("# ")
    ):
        heading, _, rest = body.lstrip().partition("\n")
        body = f"{heading}\n\n{banner}{rest}"
    else:
        body = f"# {title}\n\n{banner}{body}"
    write(out_md, body)
    return title


def render_script(script: Path, out_md: Path) -> str:
    """Render an example script as a page: docstring as prose, then the code."""
    source = script.read_text()
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree) or ""
    code = source
    if docstring and tree.body and isinstance(tree.body[0], ast.Expr):
        code = "\n".join(source.splitlines()[tree.body[0].end_lineno :]).lstrip("\n")
    # Sphinx-style roles and literals read fine as Markdown code spans.
    prose = re.sub(r":\w+:`~?([^`]+)`", r"`\1`", docstring).replace("``", "`")
    first, _, rest = prose.partition("\n")
    title = first.strip().rstrip(".") or script.stem.replace("_", " ").capitalize()
    rel = source_label(script)
    body = (
        f"# {title}\n\n"
        f'!!! note "Script"\n'
        f"    [`{rel}`]({github_url(script)}). Download it to run it yourself.\n\n"
        f"{separate_lists(rest.strip())}\n\n```python\n{code.rstrip()}\n```\n"
    )
    write(out_md, body)
    return title


def render_example(
    example: Path, out_md: Path, page_map: dict[Path, Path] | None = None
) -> str:
    if example.suffix == ".ipynb":
        return render_notebook(example, out_md)
    if example.suffix == ".md":
        assets: list[tuple[Path, Path]] = []
        content = rewrite_links(
            separate_lists(example.read_text()),
            example.parent,
            out_md.parent,
            page_map or {},
            assets,
        )
        write(out_md, content)
        for source, dest in assets:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, dest)
        return title_of(content, example.stem.replace("_", " ").capitalize())
    return render_script(example, out_md)


def example_files(directory: Path, markdown: bool = False) -> list[Path]:
    suffixes = {".ipynb", ".py"} | ({".md"} if markdown else set())
    return sorted(
        p
        for p in directory.glob("*")
        if p.suffix in suffixes and not p.name.startswith("_")
    )


def package_examples(pkg: Package) -> list[Path]:
    """Examples of a package: ``examples/``, then ``docs/examples/``."""
    return example_files(pkg.path / "examples") + example_files(
        pkg.path / "docs" / "examples", markdown=True
    )


def _module_file(src: Path, dotted: str) -> Path | None:
    base = src / Path(*dotted.split("."))
    for candidate in (base / "__init__.py", base.with_suffix(".py")):
        if candidate.is_file():
            return candidate
    return None


def resolve_definition(
    src: Path, module: str, name: str, depth: int = 0
) -> tuple[str, str]:
    """Follow imports to where ``module.name`` is defined, as Python would.

    Returns ``(dotted_path, kind)`` with kind one of function, class, attribute
    or module. A name bound in a module wins over a
    submodule of the same name, which is the case griffe gets wrong: with
    ``from .fsc import fsc`` in ``__init__``, ``pkg.fsc`` is the function.
    """
    path = _module_file(src, module)
    if path is not None and depth < 10:
        tree = ast.parse(path.read_text())
        is_pkg = path.name == "__init__.py"
        for node in tree.body:
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
            ):
                return f"{module}.{name}", "function"
            if isinstance(node, ast.ClassDef) and node.name == name:
                return f"{module}.{name}", "class"
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if any(isinstance(t, ast.Name) and t.id == name for t in targets):
                    return f"{module}.{name}", "attribute"
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if (alias.asname or alias.name) != name:
                        continue
                    if node.level:
                        parts = module.split(".")
                        parts = parts if is_pkg else parts[:-1]
                        parts = parts[: len(parts) - (node.level - 1)]
                        source = ".".join(
                            parts + ([node.module] if node.module else [])
                        )
                    else:
                        source = node.module or ""
                    if _module_file(src, source) is None:
                        return (
                            f"{source}.{alias.name}",
                            "attribute",
                        )  # outside this package
                    return resolve_definition(src, source, alias.name, depth + 1)
    if _module_file(src, f"{module}.{name}") is not None:
        return f"{module}.{name}", "module"
    return f"{module}.{name}", "attribute"


def export_targets(pkg: Package) -> list[tuple[str, str]] | None:
    """Dotted definition path and kind of each name in ``__all__``.

    Documenting the package as a whole is not enough: when a function shadows
    its own submodule (``from .fsc import fsc``), griffe documents the module.
    Without ``__all__``, the names imported into ``__init__`` from the package
    itself are used. Returns None if there are none.
    """
    init = pkg.src / pkg.module / "__init__.py"
    exports = None
    for node in ast.parse(init.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            exports = list(ast.literal_eval(node.value))
    if exports is None:
        # No __all__: the public names the package imports into __init__.
        exports = [
            alias.asname or alias.name
            for node in ast.parse(init.read_text()).body
            if isinstance(node, ast.ImportFrom) and node.level
            for alias in node.names
            if not (alias.asname or alias.name).startswith("_")
        ]
        if not exports:
            return None
    return [
        resolve_definition(pkg.src, pkg.module, name)
        for name in exports
        if name != "__version__"
    ]


def default_api_page(pkg: Package) -> str:
    header = f"# API reference\n\nEverything below is importable from `{pkg.module}` directly.\n\n"
    targets = export_targets(pkg)
    if targets is None:
        return (
            f"{header}::: {pkg.module}\n    options:\n      show_root_heading: false\n"
        )
    blocks = []
    # Entry points first: functions, then classes, modules and constants.
    order = {"function": 0, "class": 1, "module": 2, "attribute": 3}
    for path, _kind in sorted(targets, key=lambda t: order[t[1]]):
        blocks.append(
            f"::: {path}\n    options:\n      show_root_heading: true\n      show_root_full_path: false\n"
        )
    return header + "\n".join(blocks)


def generate_package(pkg: Package, page_map: dict[Path, Path]) -> None:
    assets: list[tuple[Path, Path]] = []
    readme = pkg.path / "README.md"
    text = (
        readme.read_text()
        if readme.exists()
        else f"# {pkg.name}\n\n{pkg.description}\n"
    )
    text = separate_lists(strip_badges(text))
    text = rewrite_links(text, pkg.path, pkg.out, page_map, assets)
    install = (
        f"\n## Installation\n\n```bash\npip install {pkg.name}\n```\n"
        if "pip install" not in text
        else ""
    )
    write(pkg.out / "index.md", text.rstrip() + "\n" + install)
    pkg.pages.append(("Overview", f"{pkg.category}/{pkg.name}/index.md"))

    extra_dir = pkg.path / "docs"
    custom_api = extra_dir / "api.md"
    for page in sorted(extra_dir.glob("*.md")):
        if page.name in {"api.md", "index.md"}:
            continue
        content = rewrite_links(
            separate_lists(page.read_text()), extra_dir, pkg.out, page_map, assets
        )
        write(pkg.out / page.name, content)
        pkg.pages.append(
            (title_of(content, page.stem), f"{pkg.category}/{pkg.name}/{page.name}")
        )

    api = custom_api.read_text() if custom_api.exists() else default_api_page(pkg)
    write(pkg.out / "api.md", api)
    pkg.pages.append(("API reference", f"{pkg.category}/{pkg.name}/api.md"))

    for example in package_examples(pkg):
        out_md = pkg.out / "examples" / f"{example.stem}.md"
        title = render_example(example, out_md, page_map)
        pkg.examples.append((title, out_md.relative_to(DOCS).as_posix()))

    if pkg.examples:
        base = Path(pkg.category, pkg.name)
        gallery = "\n".join(
            f"- [{t}]({Path(f).relative_to(base).as_posix()})" for t, f in pkg.examples
        )
        section = f"## Example gallery\n\n{gallery}\n\n"
        index = pkg.out / "index.md"
        text = index.read_text()
        # Before the license section if there is one, else at the end.
        match = re.search(r"^## License", text, re.MULTILINE)
        at = match.start() if match else len(text)
        index.write_text(f"{text[:at].rstrip()}\n\n{section}{text[at:]}")

    for source, dest in assets:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)


def generate_category_indexes(packages: list[Package]) -> None:
    for category, title in CATEGORIES.items():
        members = [p for p in packages if p.category == category]
        groups = list(dict.fromkeys(p.group for p in members))
        tables = []
        for group in groups:
            rows = [
                f"| [{p.name}]({p.name}/index.md) | {p.description} |"
                for p in members
                if p.group == group
            ]
            heading = f"## {group}\n\n" if group else ""
            tables.append(
                f"{heading}| Package | Description |\n| --- | --- |\n"
                + "\n".join(rows)
                + "\n"
            )
        intro_file = DOCS / "_snippets" / f"{category}.md"
        intro = intro_file.read_text().strip() + "\n\n" if intro_file.exists() else ""
        write(DOCS / category / "index.md", f"# {title}\n\n{intro}" + "\n".join(tables))


def generate_developer_docs(page_map: dict[Path, Path]) -> list[tuple[str, str]]:
    out = DOCS / "developer"
    sources = [ROOT / "CONTRIBUTING.md"] + sorted(
        p for p in (ROOT / "notes").glob("*.md") if not NOTES_SKIP.search(p.name)
    )
    nav = []
    for source in sources:
        dest = out / ("index.md" if source.name == "CONTRIBUTING.md" else source.name)
        assets: list = []
        content = rewrite_links(
            separate_lists(source.read_text()),
            source.parent,
            dest.parent,
            page_map,
            assets,
        )
        write(dest, content)
        nav.append((title_of(content, source.stem), dest.relative_to(DOCS).as_posix()))
    return nav


def nav_files(nav: object) -> set[str]:
    """Collect every page path referenced by a zensical nav structure."""
    found: set[str] = set()
    if isinstance(nav, str):
        found.add(nav)
    elif isinstance(nav, list):
        for item in nav:
            found |= nav_files(item)
    elif isinstance(nav, dict):
        for value in nav.values():
            found |= nav_files(value)
    return found


def toml_nav(packages: list[Package], developer: list) -> str:
    def q(s: str) -> str:
        return '"' + s.replace('"', '\\"') + '"'

    lines = []
    for category, title in CATEGORIES.items():
        lines.append(f"  {{ {q(title)} = [")
        lines.append(f'    "{category}/index.md",')
        for p in (p for p in packages if p.category == category):
            entries = [f"{{ {q(t)} = {q(f)} }}" for t, f in p.pages]
            if p.examples:
                ex = ", ".join(f"{{ {q(t)} = {q(f)} }}" for t, f in p.examples)
                entries.append(f'{{ "Examples" = [{ex}] }}')
            lines.append(f"    {{ {q(p.name)} = [")
            lines.extend(f"      {e}," for e in entries)
            lines.append("    ] },")
        lines.append("  ] },")
    # Hand-written placeholder until the top-level examples are curated.
    lines.append('  { "Examples" = "examples/index.md" },')
    lines.append('  { "Contributing" = [')
    lines.extend(f"    {{ {q(t)} = {q(f)} }}," for t, f in developer)
    lines.append("  ] },")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--print-nav", action="store_true", help="print generated nav entries"
    )
    parser.add_argument(
        "--strict", action="store_true", help="treat warnings as errors"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="pull the latest default branch of each I/O package clone",
    )
    args = parser.parse_args()

    for name in GENERATED_DIRS:
        shutil.rmtree(DOCS / name, ignore_errors=True)

    packages = discover_packages() + discover_io_packages(args.update)
    # Source files that become pages, so links between them survive.
    page_map: dict[Path, Path] = {
        (ROOT / "CONTRIBUTING.md").resolve(): DOCS / "developer" / "index.md",
    }
    for note in (ROOT / "notes").glob("*.md"):
        page_map[note.resolve()] = DOCS / "developer" / note.name
    for pkg in packages:
        page_map[(pkg.path / "README.md").resolve()] = pkg.out / "index.md"
        # A docs homepage is replaced by the README; other pages map one to one.
        page_map[(pkg.path / "docs" / "index.md").resolve()] = pkg.out / "index.md"
        for page in (pkg.path / "docs").glob("*.md"):
            if page.name not in {"index.md", "api.md"}:
                page_map[page.resolve()] = pkg.out / page.name
        for ex in package_examples(pkg):
            page_map[ex.resolve()] = pkg.out / "examples" / f"{ex.stem}.md"
        if example_files(pkg.path / "examples"):
            page_map[(pkg.path / "examples").resolve()] = (
                pkg.out / "index.md#example-gallery"
            )

    for pkg in packages:
        generate_package(pkg, page_map)
    generate_category_indexes(packages)
    developer = generate_developer_docs(page_map)

    if args.print_nav:
        print(toml_nav(packages, developer))
        return

    config = tomllib.loads((ROOT / "zensical.toml").read_text())
    in_nav = nav_files(config["project"].get("nav", []))
    generated = {
        p.relative_to(DOCS).as_posix()
        for name in GENERATED_DIRS
        for p in (DOCS / name).rglob("*.md")
    }
    missing = sorted(generated - in_nav)
    dangling = sorted(f for f in in_nav if not (DOCS / f).exists())
    if missing or dangling:
        for f in missing:
            print(
                f"error: generated page not in zensical.toml nav: {f}", file=sys.stderr
            )
        for f in dangling:
            print(f"error: zensical.toml nav entry has no page: {f}", file=sys.stderr)
        sys.exit("run with --print-nav for the entries to paste into zensical.toml")
    if args.strict and WARNINGS:
        sys.exit(f"{len(WARNINGS)} warning(s) in strict mode")
    print(f"generated pages for {len(packages)} packages")


if __name__ == "__main__":
    main()
