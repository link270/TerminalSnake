import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from agent import RandomAgent, begin, compile_results


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

    def test_episode_results_require_log_level_one(self):
        def finish_episode(_ep, _runs, game, _agent, _verbose):
            game.score = 2
            game.steps = 5

        with patch("agent.run_episode", side_effect=finish_episode):
            quiet = StringIO()
            with redirect_stdout(quiet):
                begin(size=3, runs=1, verbose=0, agent=RandomAgent(1), game_seed=1)
            logged = StringIO()
            with redirect_stdout(logged):
                begin(size=3, runs=1, verbose=1, agent=RandomAgent(1), game_seed=1)

        self.assertNotIn("{'episode':", quiet.getvalue())
        self.assertIn("{'episode': 0, 'score': 2, 'steps': 5}", logged.getvalue())


if __name__ == "__main__":
    unittest.main()
