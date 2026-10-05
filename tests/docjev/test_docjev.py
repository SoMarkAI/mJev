import unittest
from docjev.rlcd.metrics import probability_metrics, circular_statistics
from docjev.rlcd.prepare_docjev import components


class DocJevTests(unittest.TestCase):
    def test_paper_image_links_are_transitive(self):
        docs = [
            {"id": "a", "paper_id": "p1", "image_sha256": "h1"},
            {"id": "b", "paper_id": "p1", "image_sha256": "h2"},
            {"id": "c", "paper_id": "p2", "image_sha256": "h2"},
            {"id": "d", "paper_id": "p3", "image_sha256": "h3"},
        ]
        self.assertEqual(sorted(map(len, components(docs))), [1, 3])

    def test_accuracy_has_known_values(self):
        result = probability_metrics(
            [
                {"probabilities": [0.75, 0.25], "target_index": 0},
                {"probabilities": [0.25, 0.75], "target_index": 0},
            ]
        )
        self.assertEqual(result, {"count": 2, "accuracy": 0.5})
        self.assertEqual(
            probability_metrics([{"probabilities": [1.0, 0.0], "target_index": 0}]),
            {"count": 1, "accuracy": 1.0},
        )

    def test_consistent_wrong_decision_is_not_circular_accuracy(self):
        groups = [
            [{"semantic_prediction": 1, "semantic_target": 0}] * 3,
            [
                {"semantic_prediction": 0, "semantic_target": 0},
                {"semantic_prediction": 1, "semantic_target": 0},
            ],
        ]
        r = circular_statistics(groups)
        self.assertEqual(r["permutation_consistency"], 0.5)
        self.assertEqual(r["circular_accuracy"], 0.0)

    def test_bad_probabilities_are_rejected(self):
        with self.assertRaises(ValueError):
            probability_metrics([{"probabilities": [0.8, 0.8], "target_index": 0}])
