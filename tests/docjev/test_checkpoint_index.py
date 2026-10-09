import json
import tempfile
import unittest
from pathlib import Path
import torch
from safetensors.torch import save_file
from docjev.rlcd.checkpoint_index import inspect_index


class FullCheckpointMetadataTests(unittest.TestCase):
    def test_shard_count_is_rejected_and_corrected_using_serialized_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tensors = {
                "model.language_model.embed_tokens.weight": torch.ones(3, 2, dtype=torch.bfloat16),
                "lm_head.weight": torch.ones(3, 2, dtype=torch.bfloat16),
                "model.visual.weight": torch.ones(2, dtype=torch.bfloat16),
            }
            shard = "model-00001-of-00001.safetensors"
            save_file(tensors, root / shard)
            (root / "config.json").write_text(json.dumps({"tie_word_embeddings": True}))
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_parameters": 2, "total_size": 28},
                        "weight_map": {name: shard for name in tensors},
                    }
                )
            )
            with self.assertRaises(ValueError):
                inspect_index(root, 8)
            repaired = inspect_index(root, 8, repair=True)
            self.assertTrue(repaired["metadata_repaired"])
            self.assertEqual(repaired["stored_elements"], 14)
            self.assertEqual(repaired["unique_parameters"], 8)
            self.assertEqual(inspect_index(root, 8)["previous_total_parameters"], 8)


if __name__ == "__main__":
    unittest.main()
