"""Select a production checkout independently of the evaluation harness."""

import json
import os
import subprocess
from pathlib import Path

PRODUCTION = (".claude-plugin", "hooks", "src")


def plugin_options() -> dict[str, int | bool]:
    options: dict[str, int | bool] = {}
    diagnostics = os.environ.get("JEV_EVAL_DIAGNOSTICS", "0")
    if diagnostics not in {"0", "1"}:
        raise ValueError("JEV_EVAL_DIAGNOSTICS must be 0 or 1")
    if diagnostics == "1":
        options["diagnostics"] = True
    target = os.environ.get("JEV_EVAL_CHUNK_CHARS")
    if target is not None:
        value = int(target)
        if value < 0:
            raise ValueError("JEV_EVAL_CHUNK_CHARS must be nonnegative")
        options["chunkChars"] = value
    return options


DEFAULT_ARMS: dict[str, dict | None] = {"control": None, "plugin": {}}


def arms() -> dict[str, dict | None]:
    """Arm name -> ``None`` (no plugin) or ``{"options": {...}, "env": {VAR: "$LAUNCHER_VAR"}}``.

    ``JEV_EVAL_ARMS`` names a JSON file declaring the arms; unset keeps the historical
    control/plugin pair. Exactly one arm must be ``null`` (the native control).
    """
    path = os.environ.get("JEV_EVAL_ARMS")
    if not path:
        return dict(DEFAULT_ARMS)
    declared = json.loads(Path(path).read_text())
    if not isinstance(declared, dict) or len(declared) < 2:
        raise ValueError("JEV_EVAL_ARMS must declare at least two arms")
    controls = [name for name, value in declared.items() if value is None]
    if len(controls) != 1:
        raise ValueError("JEV_EVAL_ARMS must declare exactly one null control arm")
    for name, value in declared.items():
        if value is None:
            continue
        if not isinstance(value, dict) or set(value) - {"options", "env", "plugin"}:
            raise ValueError(
                f"Arm {name} must be null or an object with options/env/plugin"
            )
        if "plugin" in value and not isinstance(value["plugin"], str):
            raise TypeError(f"Arm {name} plugin must be an absolute checkout path")
        if not isinstance(value.get("options", {}), dict) or not isinstance(
            value.get("env", {}), dict
        ):
            raise TypeError(f"Arm {name} options/env must be objects")
    return declared


def control_arm() -> str:
    return next(name for name, value in arms().items() if value is None)


def arm_options(name: str) -> dict | None:
    """Plugin options for an arm, or ``None`` when the arm runs without the plugin."""
    value = arms()[name]
    if value is None:
        return None
    return {**plugin_options(), **value.get("options", {})}


def arm_env(name: str) -> dict[str, str]:
    """Container environment overrides; ``$NAME`` values are read from the launcher env."""
    value = arms()[name] or {}
    resolved: dict[str, str] = {}
    for key, source in value.get("env", {}).items():
        if isinstance(source, str) and source.startswith("$"):
            if not os.environ.get(source[1:]):
                raise ValueError(
                    f"Arm {name} needs {source[1:]} in the launcher environment"
                )
            resolved[key] = os.environ[source[1:]]
        else:
            resolved[key] = str(source)
    return resolved


def _checkout(value: str, what: str) -> Path:
    root = Path(value)
    if not root.is_absolute() or not all((root / name).is_dir() for name in PRODUCTION):
        raise ValueError(f"{what} must be an absolute plugin checkout")
    return root.resolve()


def production_root(repo: Path, arm: str | None = None) -> Path:
    """Plugin checkout for an arm: its ``plugin`` entry, else ``JEV_EVAL_PLUGIN_DIR``, else this repo."""
    if arm is not None:
        declared = (arms()[arm] or {}).get("plugin")
        if isinstance(declared, str) and declared.startswith("$"):
            if not os.environ.get(declared[1:]):
                raise ValueError(
                    f"Arm {arm} needs {declared[1:]} in the launcher environment"
                )
            declared = os.environ[declared[1:]]
        if declared:
            return _checkout(declared, f"Arm {arm} plugin")
    value = os.environ.get("JEV_EVAL_PLUGIN_DIR")
    if not value:
        return repo
    return _checkout(value, "JEV_EVAL_PLUGIN_DIR")


def plugin_name(root: Path) -> str:
    """The plugin's manifest name, which Claude reports in the init event and settings key."""
    manifest = json.loads((root / ".claude-plugin" / "plugin.json").read_text())
    name = manifest.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Plugin manifest under {root} has no name")
    return name


def plugin_roots(repo: Path) -> dict[str, Path]:
    """Distinct plugin checkouts used by the declared non-control arms."""
    roots: dict[str, Path] = {}
    for name, value in arms().items():
        if value is None:
            continue
        root = production_root(repo, name)
        roots.setdefault(str(root), root)
    return roots


def production_provenance(repo: Path) -> dict[str, dict[str, dict[str, str]]]:
    """Commit of every external plugin checkout in play; the harness repo is pinned elsewhere."""
    checkouts: dict[str, dict[str, str]] = {}
    for key, root in plugin_roots(repo).items():
        if root == repo.resolve():
            continue
        if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
            raise ValueError(f"Commit production checkout changes under {root} first")
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        checkouts[key] = {"production_commit": commit, "plugin": plugin_name(root)}
    return {"production_checkouts": checkouts} if checkouts else {}
