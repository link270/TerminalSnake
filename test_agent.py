import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from agent import QLearningAgent, RandomAgent, run_episodes, compile_results, run_episode
from snake import Action, SnakeGame


STATE = (False, False, False, 0, True, False, False, True)
NEXT_STATE = (False, True, False, 0, True, False, False, True)


class QLearningAgentTest(unittest.TestCase):
    def test_choose_action_creates_q_values_for_new_state(self):
        agent = QLearningAgent(seed=1, epsilon=0)

        self.assertIsInstance(agent.choose_action(STATE), Action)
        self.assertIn(STATE, agent.q_table)

    def test_learn_updates_chosen_action_value(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        agent.q_table[STATE] = [0.0, 2.0, 0.0]
        agent.q_table[NEXT_STATE] = [1.0, 4.0, 3.0]

        agent.learn(STATE, Action.STRAIGHT, reward=1, next_state=NEXT_STATE, done=False)

        self.assertAlmostEqual(agent.q_table[STATE][1], 3.3)

    def test_epsilon_decays_once_per_episode(self):
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.995)
        game = SnakeGame(render=False, seed=1)

        run_episode(0, 1, game, agent, verbose=0)

        self.assertGreater(game.steps, 1)
        self.assertEqual(agent.epsilon, 0.995)


class ResultsTest(unittest.TestCase):
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

    def test_episode_results_require_verbose_level_one(self):
        def finish_episode(_ep, _runs, game, _agent, _verbose):
            game.score = 2
            game.steps = 5

        with patch("agent.run_episode", side_effect=finish_episode):
            quiet = StringIO()
            with redirect_stdout(quiet):
                run_episodes(size=3, runs=1, verbose=0, agent=RandomAgent(1), game_seed=1)
            logged = StringIO()
            with redirect_stdout(logged):
                run_episodes(size=3, runs=1, verbose=1, agent=RandomAgent(1), game_seed=1)

        self.assertNotIn("{'episode':", quiet.getvalue())
        self.assertIn("{'episode': 0, 'score': 2, 'steps': 5}", logged.getvalue())


if __name__ == "__main__":
    unittest.main()
