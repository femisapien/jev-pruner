import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evals.full import source_hashes
from evals.sources import (
    PRODUCTION,
    plugin_name,
    plugin_options,
    plugin_roots,
    production_provenance,
    production_root,
)


class SourceTests(unittest.TestCase):
    def test_options_are_explicit_and_validated(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(plugin_options(), {})
        with patch.dict(
            os.environ,
            {"JEV_EVAL_DIAGNOSTICS": "1", "JEV_EVAL_CHUNK_CHARS": "4000"},
            clear=True,
        ):
            self.assertEqual(
                plugin_options(), {"diagnostics": True, "chunkChars": 4000}
            )
        for name, value in (
            ("JEV_EVAL_DIAGNOSTICS", "yes"),
            ("JEV_EVAL_CHUNK_CHARS", "-1"),
            ("JEV_EVAL_CHUNK_CHARS", "nan"),
        ):
            with patch.dict(os.environ, {name: value}, clear=True):
                with self.assertRaises(ValueError):
                    plugin_options()

    def test_hashes_follow_selected_production_and_local_harness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness, plugin = root / "harness", root / "plugin"
            (harness / "evals").mkdir(parents=True)
            (harness / "evals/observer.ts").write_text("observer")
            for name in PRODUCTION:
                (plugin / name).mkdir(parents=True)
            (plugin / "src/output.ts").write_text("selected production")
            (plugin / ".claude-plugin/plugin.json").write_text('{"name": "ext"}')
            with (
                patch.dict(os.environ, {"JEV_EVAL_PLUGIN_DIR": str(plugin)}),
                patch("evals.full.REPO", harness),
                patch(
                    "evals.full.subprocess.check_output",
                    side_effect=["evals/observer.ts\n", "src/output.ts\n"],
                ) as files,
            ):
                hashes = source_hashes()
            self.assertEqual(files.call_args_list[0].kwargs["cwd"], harness)
            self.assertEqual(files.call_args_list[1].kwargs["cwd"], plugin)
            self.assertEqual(
                hashes["plugin/ext/src/output.ts"],
                hashlib.sha256(b"selected production").hexdigest(),
            )
            self.assertEqual(
                hashes["evals/observer.ts"], hashlib.sha256(b"observer").hexdigest()
            )

    def test_arm_plugin_checkout_overrides_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in PRODUCTION:
                (root / name).mkdir()
            (root / ".claude-plugin/plugin.json").write_text(
                '{"name": "pruner-output"}'
            )
            declared = root / "arms.json"
            declared.write_text(
                json.dumps(
                    {
                        "native": None,
                        "jev": {"options": {}},
                        "v3": {"options": {"model": "m"}, "plugin": str(root)},
                    }
                )
            )
            with patch.dict(os.environ, {"JEV_EVAL_ARMS": str(declared)}):
                self.assertEqual(
                    production_root(Path("/harness"), "jev"), Path("/harness")
                )
                self.assertEqual(
                    production_root(Path("/harness"), "v3"), root.resolve()
                )
                self.assertEqual(plugin_name(root), "pruner-output")
                self.assertEqual(
                    set(plugin_roots(Path("/harness"))),
                    {"/harness", str(root.resolve())},
                )

    def test_rejects_relative_and_dirty_checkouts(self) -> None:
        with patch.dict(os.environ, {"JEV_EVAL_PLUGIN_DIR": "relative"}):
            with self.assertRaisesRegex(ValueError, "absolute"):
                production_root(Path("/harness"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in PRODUCTION:
                (root / name).mkdir()
            with (
                patch.dict(os.environ, {"JEV_EVAL_PLUGIN_DIR": str(root)}),
                patch(
                    "evals.sources.subprocess.check_output", return_value=b" M src/x"
                ),
            ):
                with self.assertRaisesRegex(ValueError, "Commit production"):
                    production_provenance(Path("/harness"))


if __name__ == "__main__":
    unittest.main()
