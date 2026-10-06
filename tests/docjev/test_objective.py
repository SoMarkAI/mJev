import unittest
import torch
from docjev.rlcd.objective import advantages, rlcd_loss, rewards, sample_group


class RewardTests(unittest.TestCase):
    def test_signed_weight_and_no_std_cancellation(self):
        actions = torch.tensor([0, 1, 1, 0])
        small = rewards(actions, 0, 0.2)
        large = rewards(actions, 0, 0.8)
        torch.testing.assert_close(small, torch.tensor([0.2, -0.2, -0.2, 0.2]))
        torch.testing.assert_close(advantages(large), 4 * advantages(small))

    def test_all_identical_outcomes_have_zero_advantage(self):
        for actions in [torch.zeros(8, dtype=torch.long), torch.ones(8, dtype=torch.long)]:
            self.assertEqual(advantages(rewards(actions, 0, 0.8)).abs().sum(), 0)

    def test_policy_gradient_increases_correct_candidate(self):
        logits = torch.nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))
        actions = torch.tensor([0, 1, 2, 0])
        old = logits.detach().log_softmax(-1).index_select(0, actions)
        advantage = advantages(rewards(actions, 0, 0.8))
        reference = logits.detach().log_softmax(-1)
        loss, _ = rlcd_loss(
            logits, actions, old, advantage, reference, clip_epsilon=0.2, kl_beta=0.02
        )
        loss.backward()
        self.assertLess(logits.grad[0], 0)
        self.assertGreater(logits.grad[1], 0)
        self.assertGreater(logits.grad[2], 0)

    def test_sampling_uses_candidate_softmax_without_gt(self):
        generator = torch.Generator().manual_seed(20261001)
        logits = torch.tensor([-100.0, 100.0, -100.0])
        actions, logp = sample_group(logits, 8, generator)
        self.assertTrue(torch.all(actions == 1))
        torch.testing.assert_close(logp, logits.log_softmax(-1)[actions])

    def test_exact_kl_and_detached_reference(self):
        logits = torch.tensor([1.0, -1.0], requires_grad=True)
        reference_logits = torch.tensor([0.0, 0.0], requires_grad=True)
        reference = reference_logits.log_softmax(-1)
        actions = torch.tensor([0, 1])
        old = logits.detach().log_softmax(-1)[actions]
        loss, stats = rlcd_loss(
            logits, actions, old, torch.zeros(2), reference, clip_epsilon=0.2, kl_beta=0.02
        )
        expected = (logits.softmax(-1) * (logits.log_softmax(-1) - reference)).sum()
        self.assertAlmostEqual(stats["kl"], float(expected.detach()), places=6)
        self.assertGreater(stats["kl"], 0)
        loss.backward()
        self.assertIsNone(reference_logits.grad)

    def test_clipping_uses_old_rollout_policy(self):
        logits = torch.tensor([2.0, -2.0], requires_grad=True)
        actions = torch.tensor([0])
        old = torch.tensor([-0.69314718])
        loss, _ = rlcd_loss(
            logits,
            actions,
            old,
            torch.ones(1),
            torch.zeros(2).log_softmax(-1),
            clip_epsilon=0.2,
            kl_beta=0.0,
        )
        self.assertAlmostEqual(float(loss.detach()), -1.2, places=5)
        loss.backward()
        torch.testing.assert_close(logits.grad, torch.zeros_like(logits))

    def test_invalid_target_probability_is_rejected(self):
        for value in [-0.1, 1.1, float("nan")]:
            with self.assertRaises(ValueError):
                rewards(torch.tensor([0, 1]), 0, value)


if __name__ == "__main__":
    unittest.main()
