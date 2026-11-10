"""Every third-party package that the repository's Python files import directly must be declared.

Declared means: listed in a root `[dependency-groups]` group or in `core/pyproject.toml` or
`ml/pyproject.toml`. Relying on
a package arriving indirectly through another one can break a clean setup (CI, a fresh clone).
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from importlib.metadata import packages_distributions
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIRST_PARTY = {"defect_core", "defect_ml"}  # our own packages
# Scripts import their siblings by file name (scripts/data/*.py); these are not packages.
LOCAL_MODULES = {p.stem for p in (REPO / "scripts").rglob("*.py")}


def normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(spec: str) -> str:
    return normalise(re.split(r"[\[<>=!~; ]", spec.strip())[0])


def declared_packages() -> set[str]:
    root = tomllib.loads((REPO / "pyproject.toml").read_text())
    specs = [s for group in root.get("dependency-groups", {}).values() for s in group]
    for package in ("core", "ml"):
        project = tomllib.loads((REPO / package / "pyproject.toml").read_text())
        specs += project["project"].get("dependencies", [])
    return {requirement_name(s) for s in specs}


def imported_top_level_modules() -> dict[str, list[str]]:
    found: dict[str, list[str]] = {}
    for base in ("core", "ml", "scripts"):
        for path in (REPO / base).rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    modules = [a.name.split(".")[0] for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module.split(".")[0]]
                else:
                    continue
                for module in modules:
                    found.setdefault(module, []).append(str(path.relative_to(REPO)))
    return found


def test_directly_imported_packages_are_declared() -> None:
    declared = declared_packages()
    distributions = packages_distributions()
    missing = {}
    for module, files in imported_top_level_modules().items():
        if module in sys.stdlib_module_names or module in FIRST_PARTY | LOCAL_MODULES:
            continue
        owners = {normalise(d) for d in distributions.get(module, [])}
        assert owners, f"cannot find an installed package that provides '{module}' ({files[0]})"
        if not owners & declared:
            missing[module] = (sorted(owners), sorted(set(files))[:2])
    assert not missing, (
        "imported directly but not declared in any pyproject.toml dependency list "
        f"(import name: (package, example files)): {missing}"
    )
