import unittest
from docjev.rlcd.distributed_eval import evaluation_schedule


class CollectiveEvaluationTest(unittest.TestCase):
    def test_equal_collectives_and_exact_coverage(self):
        for size in [1, 2, 4, 5, 22, 24]:
            rows = list(range(size))
            schedules = [list(evaluation_schedule(rows, rank, 4)) for rank in range(4)]
            self.assertEqual(len({len(schedule) for schedule in schedules}), 1)
            kept = [row for schedule in schedules for row, retain in schedule if retain]
            self.assertEqual(sorted(kept), rows)

    def test_invalid_inputs(self):
        for rows, rank, world in [([], 0, 4), ([1], 4, 4), ([1], 0, 0)]:
            with self.assertRaises(ValueError):
                list(evaluation_schedule(rows, rank, world))
