"""Deadline selection must never use late or subsequently mutated outputs."""
import unittest
from unittest.mock import patch
import torch
from baseline_sr.time_budget import TimeBudget


class BudgetChecks(unittest.TestCase):
    def test_independent_deadlines_and_snapshot_immutability(self):
        clock = [0.]
        with patch('baseline_sr.time_budget.time.perf_counter', side_effect=lambda: clock[0]):
            budget = TimeBudget({'optimization': 10., 'end_to_end': 12.}, 'cpu')
            value = torch.tensor([1.])
            clock[0] = 9.
            self.assertFalse(budget.observe(value, steps=1))
            value.fill_(2.)
            clock[0] = 11.
            self.assertFalse(budget.observe(value, steps=2))
            value.fill_(3.)
            clock[0] = 13.
            self.assertTrue(budget.observe(value, steps=3))
            self.assertEqual(budget.snapshots['optimization']['steps'], 1)
            self.assertEqual(budget.snapshots['end_to_end']['steps'], 2)
            self.assertEqual(budget.snapshots['optimization']['reconstruction'].item(), 1.)
            self.assertEqual(budget.snapshots['end_to_end']['reconstruction'].item(), 2.)

    def test_copy_crossing_deadline_is_rejected(self):
        with patch('baseline_sr.time_budget.time.perf_counter', side_effect=[0., 9., 11.]):
            budget = TimeBudget({'limit': 10.}, 'cpu')
            self.assertTrue(budget.observe(torch.ones(1), steps=1))
            self.assertEqual(budget.snapshots, {})


if __name__ == '__main__':
    unittest.main()
