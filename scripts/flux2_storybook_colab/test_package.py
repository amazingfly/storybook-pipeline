#!/usr/bin/env python3
"""Structural checks that do not require a GPU or model download."""

from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class PackageTests(unittest.TestCase):
    def test_configuration_and_manifest(self) -> None:
        config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
        manifest = json.loads((HERE / "test_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(config["base_model"], "black-forest-labs/FLUX.2-klein-base-4B")
        self.assertEqual(config["trigger"], "LQK4N")
        self.assertEqual(config["memory_mode"], "text_encoder_4bit")
        self.assertTrue(config["stop_session_after_recovery"])
        self.assertEqual((config["width"], config["height"]), (832, 1216))
        self.assertEqual(len(manifest["jobs"]), 6)
        self.assertEqual({job["scene"] for job in manifest["jobs"]}, {2, 88, 92})
        self.assertEqual({job["lora_scale"] for job in manifest["jobs"]}, {0.8, 1.0})
        self.assertEqual(len({job["id"] for job in manifest["jobs"]}), 6)

    def test_remote_runner_is_resumable_and_t4_specific(self) -> None:
        source = (HERE / "remote_generate.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        self.assertTrue(any(isinstance(node, ast.FunctionDef) and node.name == "valid_image" for node in tree.body))
        self.assertIn('"T4" not in name.upper()', source)
        self.assertIn("temporary.replace(output)", source)
        self.assertIn("save_archive(remote_root)", source)
        self.assertIn("load_in_4bit=True", source)
        self.assertIn("AutoModelForCausalLM.from_pretrained", source)
        self.assertGreaterEqual(source.count('device_map="cuda"'), 2)
        self.assertNotIn("except BaseException", source)
        self.assertNotIn("raise SystemExit(main())", source)
        self.assertIn("All {len(jobs)} jobs already complete", source)

    def test_bootstrap_removes_incompatible_optional_torchao(self) -> None:
        source = (HERE / "bootstrap.py").read_text(encoding="utf-8")
        self.assertIn('"uninstall", "--yes", "--quiet", "torchao"', source)

    def test_launcher_treats_remote_failure_as_failure(self) -> None:
        source = (HERE / "run_test.sh").read_text(encoding="utf-8")
        self.assertIn('results/failure.json', source)
        self.assertIn("GENERATE_RC=1", source)
        self.assertIn('stop -s "$SESSION"', source)


if __name__ == "__main__":
    unittest.main()
