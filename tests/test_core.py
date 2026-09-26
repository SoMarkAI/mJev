import unittest
import torch
from mjev.patch import visibility, batch_route
from mjev.decision import choose

class CoreTests(unittest.TestCase):
    def test_isolation_and_answer(self):
        mask = visibility(torch, torch.arange(10), 10, [[3,5],[5,8]])
        self.assertFalse(mask[5:8,3:5].any())
        self.assertTrue(mask[7,:3].all())
        self.assertTrue(mask[7,5:8].all())
        self.assertTrue(mask[9,:].all())
        self.assertFalse(mask[:3,3:].any())
        self.assertTrue(visibility(torch, torch.arange(10), 10, [[3,5],[5,8]], 'causal')[7,3])
    def test_batch_guard(self):
        with self.assertRaises(RuntimeError):
            batch_route([{'mode':'stock'},{'mode':'isolated'}])
    def test_tie_and_nan(self):
        self.assertEqual(choose([{'label':'A','raw_logit':2.0},{'label':'B','raw_logit':2.0}])['tied_labels'],['A','B'])
        with self.assertRaises(ValueError):
            choose([{'label':'A','raw_logit':float('nan')}])

if __name__ == '__main__':
    unittest.main()
