"""
Guards the one deploy failure this project keeps re-hitting: a new
top-level module or package that nothing adds to the Dockerfile's COPY
lines.

The flat file-list COPY silently skips anything not named in it -- no
build error -- so the image builds "successfully" and the container
crash-loops at startup on ModuleNotFoundError. Three occurrences on
record, which is what makes this worth a test rather than another
comment: news_adapters/ (2026-09-02, caught by deploy-engineer before
INT), telemetry_providers/ (caught in code review before a build), and
jev_client.py (2026-09-24, caught while wiring a deploy -- the Dockerfile
comment added after the first two did not prevent the third).

Walks the real import graph from the container's own entrypoints rather
than hardcoding a list, so it stays correct as modules come and go.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# What `docker run ... myfirstagent-bot` actually starts, plus the
# single-bot entrypoint kept working alongside it.
ENTRYPOINTS = ["combined_bot.py", "bot.py"]


def _local_module_name(name: str) -> str | None:
    """The repo-root module/package `name` refers to, or None if it's a
    third-party or stdlib import. Only the first segment matters --
    `from storage.sqlite import x` is covered by copying `storage/`."""
    root = name.split(".")[0]
    if (REPO_ROOT / f"{root}.py").is_file():
        return f"{root}.py"
    if (REPO_ROOT / root / "__init__.py").is_file():
        return root
    return None


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


def _reachable_local_modules() -> set[str]:
    """Every repo-root module/package reachable from the entrypoints,
    transitively -- what the image actually has to contain."""
    seen: set[str] = set()
    queue = list(ENTRYPOINTS)
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        path = REPO_ROOT / current
        # A package: walk its own .py files too, since any of them may
        # pull in another repo-root module.
        sources = [path] if path.is_file() else sorted(path.rglob("*.py"))
        for source in sources:
            for imported in _imports_of(source):
                local = _local_module_name(imported)
                if local is not None and local not in seen:
                    queue.append(local)
    return seen


def _dockerfile_copied_sources() -> set[str]:
    copied: set[str] = set()
    for line in (REPO_ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("COPY "):
            continue
        # Drop `COPY`, any --flags, and the trailing destination.
        tokens = [t for t in stripped.split()[1:] if not t.startswith("--")]
        copied.update(tokens[:-1])
    return copied


def test_every_module_the_entrypoints_import_is_copied_into_the_image():
    missing = sorted(_reachable_local_modules() - _dockerfile_copied_sources())
    assert not missing, (
        "These repo-root modules/packages are imported (transitively) by the "
        f"container's entrypoints but are not in any Dockerfile COPY line: {missing}. "
        "The flat file-list COPY skips unnamed files silently, so the image would "
        "build fine and the container would crash-loop at startup on "
        "ModuleNotFoundError. Add each to the file list (a module) or give it its "
        "own COPY line (a package -- COPY needs its own destination for a directory)."
    )


def test_the_guard_itself_resolves_a_known_module_and_package():
    """Cheap sanity check on the walker: if _local_module_name or the
    import walk silently stopped resolving anything, the test above would
    pass vacuously and guard nothing."""
    reachable = _reachable_local_modules()
    assert "guardrails.py" in reachable      # a plain module
    assert "news_adapters" in reachable      # a package, reached via news_sources
    assert "jev_client.py" in reachable      # reached only through guardrails
