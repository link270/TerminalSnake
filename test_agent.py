import unittest

from agent import compile_results


class CompileResultsTest(unittest.TestCase):
    def test_compiles_episode_results(self):
        results = [
            {"episode": 0, "score": 1, "steps": 10},
            {"episode": 1, "score": 3, "steps": 8},
            {"episode": 2, "score": 2, "steps": 10},
        ]

        self.assertEqual(compile_results(results), {
            "average_score": 2,
            "best_score": 3,
            "best_score_episode": 1,
            "average_survival": 28 / 3,
            "best_survival": 10,
            "best_survival_episode": 0,
        })


if __name__ == "__main__":
    unittest.main()
