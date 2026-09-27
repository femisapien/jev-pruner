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
        if not isinstance(value, dict) or set(value) - {"options", "env"}:
            raise ValueError(f"Arm {name} must be null or an object with options/env")
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


def production_root(repo: Path) -> Path:
    value = os.environ.get("JEV_EVAL_PLUGIN_DIR")
    if not value:
        return repo
    root = Path(value)
    if not root.is_absolute() or not all((root / name).is_dir() for name in PRODUCTION):
        raise ValueError("JEV_EVAL_PLUGIN_DIR must be an absolute plugin checkout")
    return root.resolve()


def production_provenance(repo: Path) -> dict[str, str]:
    if not os.environ.get("JEV_EVAL_PLUGIN_DIR"):
        return {}
    root = production_root(repo)
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise ValueError("Commit production checkout changes before execution")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    return {"production_checkout": str(root), "production_commit": commit}
