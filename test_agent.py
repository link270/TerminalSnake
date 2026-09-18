import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from agent import QLearningAgent, RandomAgent, begin, compile_results
from snake import Action


class CompileResultsTest(unittest.TestCase):
    def test_q_learning_agent_can_choose_an_action(self):
        agent = QLearningAgent(seed=1, epsilon=0)
        state = (False, False, False, 0, True, False, False, True)

        self.assertIsInstance(agent.choose_action(state), Action)
        self.assertIn(state, agent.q_table)

    def test_q_learning_agent_can_learn(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        state = (False, False, False, 0, True, False, False, True)
        next_state = (False, True, False, 0, True, False, False, True)
        agent.q_table[state] = [0.0, 2.0, 0.0]
        agent.q_table[next_state] = [1.0, 4.0, 3.0]

        agent.learn(state, Action.STRAIGHT, reward=1, next_state=next_state, done=False)

        self.assertAlmostEqual(agent.q_table[state][1], 3.3)

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
