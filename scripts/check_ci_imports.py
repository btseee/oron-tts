"""Check that no test module needs, at import time, something CI does not install.

CI installs no torch and no F5-TTS: `oron_tts.text` is pure stdlib by design and
the text tests are the majority, so a 2 GB install would only make CI slow
enough to be switched off. That holds only while nothing is imported at *module
scope* along the path from a test file to the code it exercises -- and a
collection error is a hard failure, not a skip.

This is static rather than a simulated run, because absence cannot be simulated
faithfully: a missing module makes `import x` raise but
`importlib.util.find_spec("x")` return None, and no meta-path hook does both.
Walking the import graph has no such ambiguity.

    python scripts/check_ci_imports.py

It reports the chain, so the fix is obvious: move the import inside the function
that needs it, the way `eval/metrics.py` defers torch.
"""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# What the workflow's `pip install` line provides, plus the standard library.
# Keep in step with .github/workflows/test.yml.
CI_PROVIDES = {"pytest", "ruff", "numpy", "librosa", "soundfile", "yaml", "oron_tts"}

_STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _module_level_imports(path: Path, package: str) -> set[str]:
    """Top-level imports only: anything inside a function is deferred and fine.

    Full dotted names, not just the first component. Truncating to the package
    root stops the walk at `oron_tts/__init__.py` and misses anything a
    submodule imports -- which is most of the code a test actually reaches.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in tree.body:                      # body, not walk: top level only
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # Relative imports anchor on the containing *package*, not the
                # module: `from .constants import X` in pipeline/corpus.py is
                # pipeline.constants, not pipeline.corpus.constants.
                base = package
                for _ in range(node.level - 1):
                    base = base.rpartition(".")[0]
                found.add(f"{base}.{node.module}" if node.module else base)
            elif node.module:
                found.add(node.module)
                # `from oron_tts.text import numbers` reaches the submodule, but
                # `from build_f5_dataset import select_splits` imports a
                # function -- so a dotted candidate counts only if it resolves
                # to a file.
                for alias in node.names:
                    dotted = f"{node.module}.{alias.name}"
                    if _local_path(dotted) is not None:
                        found.add(dotted)
    return found


def _guarded(path: Path) -> set[str]:
    """Modules the file skips itself over with `pytest.importorskip`.

    At module scope that skips the whole file at collection, so anything it
    names -- and anything reached only through it -- cannot fail CI.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "importorskip"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value.split(".")[0])
    return names


def _local_path(module: str) -> Path | None:
    """Where a first-party module lives, if it is one."""
    rel = module.replace(".", "/")
    for candidate in (ROOT / f"{rel}.py",
                      ROOT / rel / "__init__.py",
                      ROOT / "scripts" / f"{module}.py"):
        if candidate.exists():
            return candidate
    return None


def _walk(path: Path, package: str, chain: list[str], seen: set[Path],
          guarded: frozenset[str]) -> list[tuple[str, list[str]]]:
    if path in seen:
        return []
    seen.add(path)
    problems: list[tuple[str, list[str]]] = []
    for name in sorted(_module_level_imports(path, package)):
        root = name.split(".")[0]
        if root in _STDLIB or root in guarded:
            continue
        local = _local_path(name)
        if local is not None:
            # The package a relative import inside that file will anchor on.
            child = name if local.name == "__init__.py" else name.rpartition(".")[0]
            problems += _walk(local, child, [*chain, name], seen, guarded)
        elif root not in CI_PROVIDES:
            problems.append((root, [*chain, name]))
    return problems


def main() -> int:
    problems: dict[str, list[str]] = {}
    for test in sorted((ROOT / "tests").glob("test_*.py")):
        guarded = frozenset(_guarded(test))
        for name, chain in _walk(test, "", [test.name], set(), guarded):
            problems.setdefault(f"{test.name}: {name}", chain)

    if not problems:
        print(f"Every test module collects with only: {', '.join(sorted(CI_PROVIDES))}")
        return 0

    print("These are imported at module scope but not installed in CI:\n")
    for label, chain in sorted(problems.items()):
        print(f"  {label}\n      via {' -> '.join(chain)}")
    print("\nMove the import inside the function that needs it, or guard the test "
          "module with pytest.importorskip.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
