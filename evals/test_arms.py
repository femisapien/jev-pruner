import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from evals.full import aggregate, validate_manifest
from evals.sources import arm_env, arm_options, arms, control_arm

ARMS = {
    "native": None,
    "jev": {},
    "pruner-v2": {
        "options": {
            "baseUrl": "https://scorer.example/v1/systemone",
            "model": "stage-c-v2-001-e4",
            "keepThreshold": 0.05,
        },
        "env": {"TYPESAFE_API_KEY": "$PRUNER_SERVE_TOKEN"},
    },
}


class ArmsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name) / "arms.json"
        self.path.write_text(json.dumps(ARMS))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_default_is_the_historical_control_plugin_pair(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("JEV_EVAL_ARMS", None)
            os.environ.pop("JEV_EVAL_CHUNK_CHARS", None)
            self.assertEqual(arms(), {"control": None, "plugin": {}})
            self.assertEqual(control_arm(), "control")
            self.assertIsNone(arm_options("control"))
            self.assertEqual(arm_options("plugin"), {})
            self.assertEqual(arm_env("plugin"), {})

    def test_declared_arms_resolve_options_and_launcher_env(self) -> None:
        env = {
            "JEV_EVAL_ARMS": str(self.path),
            "JEV_EVAL_CHUNK_CHARS": "800",
            "PRUNER_SERVE_TOKEN": "tok",
        }
        with mock.patch.dict(os.environ, env):
            self.assertEqual(control_arm(), "native")
            self.assertIsNone(arm_options("native"))
            self.assertEqual(arm_options("jev"), {"chunkChars": 800})
            self.assertEqual(
                arm_options("pruner-v2"),
                {
                    "chunkChars": 800,
                    "baseUrl": "https://scorer.example/v1/systemone",
                    "model": "stage-c-v2-001-e4",
                    "keepThreshold": 0.05,
                },
            )
            self.assertEqual(arm_env("pruner-v2"), {"TYPESAFE_API_KEY": "tok"})
            self.assertEqual(arm_env("native"), {})
        with mock.patch.dict(os.environ, {"JEV_EVAL_ARMS": str(self.path)}):
            os.environ.pop("PRUNER_SERVE_TOKEN", None)
            with self.assertRaises(ValueError):
                arm_env("pruner-v2")

    def test_declaration_requires_exactly_one_control(self) -> None:
        for bad in (
            {"a": None},
            {"a": None, "b": None},
            {"a": {}, "b": {}},
            {"a": None, "b": {"x": 1}},
        ):
            self.path.write_text(json.dumps(bad))
            with (
                mock.patch.dict(os.environ, {"JEV_EVAL_ARMS": str(self.path)}),
                self.assertRaises((ValueError, TypeError)),
            ):
                arms()

    def test_manifest_and_pairs_cover_every_declared_arm(self) -> None:
        with mock.patch.dict(os.environ, {"JEV_EVAL_ARMS": str(self.path)}):
            os.environ.pop("JEV_EVAL_CHUNK_CHARS", None)
            manifest = [
                {
                    "task": task,
                    "arm": arm,
                    "job_name": f"{task}-{arm}",
                    "state": "finished",
                    "reward": reward,
                }
                for task in ("one", "two")
                for arm, reward in (("native", 1), ("jev", 0), ("pruner-v2", 1))
            ]
            validate_manifest(manifest, task_count=2)
            with self.assertRaises(ValueError):
                validate_manifest(
                    [row for row in manifest if row["arm"] != "jev"], task_count=2
                )
            result = aggregate(manifest)
            self.assertEqual(result["control_arm"], "native")
            self.assertEqual(
                sorted(result["aggregate"]), ["jev", "native", "pruner-v2"]
            )
            self.assertEqual(
                [
                    (pair["task"], pair["arm"], pair["disagreement"])
                    for pair in result["pairs"]
                ],
                [
                    ("one", "jev", True),
                    ("one", "pruner-v2", False),
                    ("two", "jev", True),
                    ("two", "pruner-v2", False),
                ],
            )
            self.assertEqual(result["arms"]["pruner-v2"]["model"], "stage-c-v2-001-e4")
