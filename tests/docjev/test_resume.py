import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from docjev.rlcd.common import write_json, sha256
from docjev.rlcd.resume import inspect_resume, layout, restore_rank


class ResumeTests(unittest.TestCase):
    def test_adam_and_sampling_restore_match_uninterrupted_next_update(self):
        torch.manual_seed(7)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        sampling = torch.Generator().manual_seed(91)
        x = torch.ones(1, 2)
        for _ in range(4):
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()
        with tempfile.TemporaryDirectory() as directory:
            saved_weights = {k: v.clone() for k, v in model.state_dict().items()}
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "layout": layout(model),
                    "sampling_rng": sampling.get_state(),
                    "torch_cpu_rng": torch.get_rng_state(),
                    "torch_cuda_rng": torch.get_rng_state(),
                    "step": 4,
                },
                Path(directory) / "rank0.pt",
            )
            expected_actions = torch.multinomial(
                torch.tensor([0.25, 0.75]), 8, replacement=True, generator=sampling
            )
            optimizer.zero_grad()
            model(x).sum().backward()
            optimizer.step()
            expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
            clone = torch.nn.Linear(2, 2)
            clone.load_state_dict(saved_weights)
            restored = torch.optim.AdamW(clone.parameters(), lr=0.01)
            rng = torch.Generator().manual_seed(1234)
            with (
                patch("docjev.rlcd.resume.dist.get_rank", return_value=0),
                patch("docjev.rlcd.resume.torch.cuda.set_rng_state"),
            ):
                self.assertEqual(restore_rank(directory, clone, restored, rng), 4)
            torch.testing.assert_close(
                torch.multinomial(torch.tensor([0.25, 0.75]), 8, replacement=True, generator=rng),
                expected_actions,
            )
            restored.zero_grad()
            clone(x).sum().backward()
            restored.step()
            for k, v in clone.state_dict().items():
                torch.testing.assert_close(v, expected_weights[k], rtol=0, atol=0)

    def test_resume_contract_and_modified_artifact_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)
            (p / "rank0.pt").write_bytes(b"checkpoint")
            write_json(
                p / "manifest.json",
                {
                    "step": 4,
                    "contract": {"world": 4},
                    "files": {"rank0.pt": sha256(p / "rank0.pt")},
                },
            )
            (p / "_SUCCESS").write_text("saved")
            self.assertEqual(inspect_resume(p, {"world": 4})["step"], 4)
            with self.assertRaises(ValueError):
                inspect_resume(p, {"world": 2})
            (p / "rank0.pt").write_bytes(b"changed")
            with self.assertRaises(ValueError):
                inspect_resume(p, {"world": 4})
