import unittest

from evals.cohort_manifest import select


class SelectTest(unittest.TestCase):
    def test_stride_selection_is_deterministic(self):
        names = [f"t{i:02d}" for i in range(10)]
        self.assertEqual(select(names, 3, 3, 1), ["t01", "t04", "t07"])

    def test_short_pool_is_rejected(self):
        with self.assertRaises(ValueError):
            select(["a", "b"], 3, 1, 0)


if __name__ == "__main__":
    unittest.main()
