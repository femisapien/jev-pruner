"""Predeclare a multi-arm paired cohort manifest for ``evals.full``.

Task selection is a written rule, not a hand-pick: eligible tasks are those whose
declared resources fit a small sandbox (``--max-cpus``, ``--max-memory-gb``, no GPU)
and whose agent timeout is at most ``--max-agent-seconds``; from the alphabetical
list of eligible tasks every ``--stride``-th task from ``--offset`` is taken until
``--task-count`` is reached. Arm order rotates by task index so every arm leads
equally often; repetitions are ordered after one another so the first repetition
completes before the second starts.
"""

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import tomllib

from evals.sources import arms


def eligible_tasks(
    source: Path, max_cpus: int, max_memory_gb: int, max_agent_seconds: float
) -> list[str]:
    names = []
    for config_path in sorted(source.glob("*/task.toml")):
        config = tomllib.loads(config_path.read_text())
        env = config["environment"]
        memory = str(env.get("memory", "0G"))
        if not memory.endswith("G") or not memory[:-1].isdigit():
            continue
        if (
            (env.get("cpus") or 1) <= max_cpus
            and int(memory[:-1]) <= max_memory_gb
            and not env.get("gpus")
            and config["agent"]["timeout_sec"] <= max_agent_seconds
        ):
            names.append(config_path.parent.name)
    return names


def select(names: list[str], count: int, stride: int, offset: int) -> list[str]:
    chosen = names[offset::stride][:count]
    if len(chosen) != count:
        raise ValueError(f"Only {len(chosen)} tasks satisfy the selection rule")
    return chosen


def build(
    source: Path, tasks: list[str], names: list[str], repetitions: int
) -> list[dict]:
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
        raise ValueError("Task checkout must be clean")
    prefix = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "--show-prefix"], text=True
    ).strip()
    rows = []
    for repetition in range(1, repetitions + 1):
        for index, task in enumerate(tasks):
            tree = subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", f"{head}:{prefix}{task}"],
                text=True,
            ).strip()
            start = (index + repetition - 1) % len(names)
            for arm in names[start:] + names[:start]:
                rows.append(
                    {
                        "task": task,
                        "arm": arm,
                        "repetition": repetition,
                        "job_name": f"{task}-r{repetition}-{arm}",
                        "resource_blocked": False,
                        "git_commit_id": head,
                        "git_tree": tree,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--benchmark-source", type=Path, required=True)
    parser.add_argument("--arms", type=Path, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-cpus", type=int, default=2)
    parser.add_argument("--max-memory-gb", type=int, default=4)
    parser.add_argument("--max-agent-seconds", type=float, default=1800)
    args = parser.parse_args()
    if (args.evidence / "manifest.json").exists():
        raise ValueError("Refusing to overwrite a declared manifest")
    os.environ["JEV_EVAL_ARMS"] = str(args.arms.resolve())
    names = list(arms())
    source = args.benchmark_source.resolve()
    pool = eligible_tasks(
        source, args.max_cpus, args.max_memory_gb, args.max_agent_seconds
    )
    tasks = select(pool, args.task_count, args.stride, args.offset)
    rows = build(source, tasks, names, args.repetitions)
    args.evidence.mkdir(parents=True, exist_ok=True)
    (args.evidence / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    protocol = {
        "declared_at": datetime.now(UTC).isoformat(),
        "arms_file": str(args.arms.resolve()),
        "arms_sha256": hashlib.sha256(args.arms.read_bytes()).hexdigest(),
        "arms": names,
        "selection_rule": {
            "max_cpus": args.max_cpus,
            "max_memory_gb": args.max_memory_gb,
            "no_gpu": True,
            "max_agent_seconds": args.max_agent_seconds,
            "order": "alphabetical",
            "stride": args.stride,
            "offset": args.offset,
            "task_count": args.task_count,
        },
        "eligible_tasks": pool,
        "tasks": tasks,
        "repetitions": args.repetitions,
        "arm_order": "rotate start arm by (task index + repetition - 1)",
        "trials": len(rows),
    }
    (args.evidence / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print(f"{len(tasks)} tasks x {len(names)} arms x {args.repetitions} = {len(rows)}")


if __name__ == "__main__":
    main()
