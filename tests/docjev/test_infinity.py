import math
import unittest
import torch
from docjev.rlcd.prepare_infinity import exact_validation_groups
from docjev.rlcd.distributed_eval import training_position
from docjev.rlcd.objective import rlcd_loss


class InfinityTrainingTests(unittest.TestCase):
    def test_exact_250_keeps_whole_groups_including_connected_images(self):
        sizes = [8] * 30 + [3, 5, 2, 7]
        groups = [[{"retained_pairs": list(range(n))}] for n in sizes]
        # A connected multi-page document remains an indivisible group.
        groups.append([{"retained_pairs": [1, 2]}, {"retained_pairs": [3, 4]}])
        selected = exact_validation_groups(groups, 250)
        self.assertEqual(sum(len(d["retained_pairs"]) for i in selected for d in groups[i]), 250)
        self.assertEqual(set(range(len(groups))), selected | (set(range(len(groups))) - selected))

    def test_impossible_exact_group_count_fails_without_splitting_images(self):
        with self.assertRaises(ValueError):
            exact_validation_groups([[{"retained_pairs": list(range(6))}]] * 100, 500)

    def test_eight_rank_first_epoch_covers_real_rows_once_without_wraparound(self):
        for count in (9, 500, 15158):
            positions = [
                training_position(count, step, rank, 8)
                for step in range(math.ceil(count / 8))
                for rank in range(8)
            ]
            self.assertEqual(sorted(i for i, pad in positions if not pad), list(range(count)))
            self.assertEqual(sum(pad for i, pad in positions), (-count) % 8)

    def test_collective_padding_has_no_policy_or_kl_gradient(self):
        logits = torch.tensor([1.0, -1.0], requires_grad=True)
        actions = torch.tensor([0, 1])
        loss, _ = rlcd_loss(
            logits,
            actions,
            torch.tensor([-0.7, -0.7]),
            torch.tensor([1.0, -1.0]),
            torch.zeros(2).log_softmax(-1),
            clip_epsilon=0.2,
            kl_beta=0.02,
        )
        (loss * 0.0).backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))
