import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.finalize_storybook_accessories_v5 import finalize


class FinalizeStorybookAccessoriesV5Test(unittest.TestCase):
    def test_materializes_only_explicitly_accepted_variant(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "wand_only").mkdir()
            (root / "grip_repaired").mkdir()
            (root / "review").mkdir()
            Image.new("RGB", (32, 48), "red").save(
                root / "wand_only" / "candidate_01.png"
            )
            Image.new("RGB", (32, 48), "green").save(
                root / "grip_repaired" / "candidate_01.png"
            )
            summary = {
                "target_accepted": 1,
                "status": "awaiting_visual_review",
                "records": [
                    {
                        "id": "candidate_01",
                        "artifacts": {
                            "wand_only": "wand_only/candidate_01.png",
                            "grip_repaired": "grip_repaired/candidate_01.png",
                        },
                    }
                ],
            }
            (root / "summary.json").write_text(json.dumps(summary))
            review = {
                "records": [
                    {
                        "id": "candidate_01",
                        "decision": "accept",
                        "variant": "wand_only",
                    }
                ]
            }
            review_path = root / "review" / "human_review.json"
            review_path.write_text(json.dumps(review))

            manifest = finalize(root, review_path)

            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["selected_count"], 1)
            self.assertEqual(manifest["records"][0]["variant"], "wand_only")
            self.assertTrue(
                (root / manifest["records"][0]["destination"]).is_file()
            )
            updated = json.loads((root / "summary.json").read_text())
            self.assertEqual(updated["records"][0]["selected_variant"], "wand_only")


if __name__ == "__main__":
    unittest.main()
