import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch
import random
import torch

from agent import QLearningAgent, RandomAgent, DQNAgent, run_episodes, compile_results, run_episode
from snake import Action, SnakeGame, StepResult
from collections import deque


STATE = (False, False, False, 0, True, False, False, True)
NEXT_STATE = (False, True, False, 0, True, False, False, True)


class QLearningAgentTest(unittest.TestCase):
    def test_choose_action_does_not_mutate_unknown_state(self):
        agent = QLearningAgent(seed=1, epsilon=0)

        self.assertIsInstance(agent.choose_action(STATE), Action)
        self.assertNotIn(STATE, agent.q_table)

    def test_learn_updates_chosen_action_value(self):
        agent = QLearningAgent(alpha=0.5, gamma=0.9)
        agent.q_table[STATE] = [0.0, 2.0, 0.0]
        agent.q_table[NEXT_STATE] = [1.0, 4.0, 3.0]

        agent.learn(STATE, Action.STRAIGHT, reward=1, next_state=NEXT_STATE, done=False)

        self.assertAlmostEqual(agent.q_table[STATE][1], 3.3)

    def test_epsilon_decays_once_per_episode(self):
        agent = QLearningAgent(epsilon=1.0, epsilon_decay=0.995)
        game = SnakeGame(render=False, seed=1)

        run_episode(0, 1, game, agent, verbose=0, train=True)

        self.assertGreater(game.steps, 1)
        self.assertEqual(agent.epsilon, 0.995)

    def test_epsilon_decays_after_food_limit(self):
        agent = QLearningAgent(seed=1)
        game = SnakeGame(size=3, render=False, seed=1)
        with patch.object(game, "step", return_value=StepResult(game.observation, 0, False)) as step:
            with redirect_stdout(StringIO()):
                outcome = run_episode(0, 1, game, agent, verbose=0, train=True)

        self.assertTrue(outcome["truncated"])
        self.assertEqual(step.call_count, 2 * game.size * game.size)
        self.assertEqual(agent.epsilon, 0.995)

class DQNAgentTest(unittest.TestCase):
    def test_agent_seed_test(self):
        seed = random.randrange(100000)
        agent_a = DQNAgent(seed=seed)
        agent_b = DQNAgent(seed=seed)

        for param_a, param_b in zip(
            agent_a.model.parameters(),
            agent_b.model.parameters()
        ):
            self.assertTrue(torch.equal(param_a, param_b))

    def test_replay_retains_transitions_and_evicts_oldest(self):
        agent = DQNAgent(seed=1)
        agent.memory = deque(maxlen=2)

        first = (STATE, Action.LEFT, 1, NEXT_STATE, False)
        second = (NEXT_STATE, Action.STRAIGHT, 0, STATE, False)
        third = (STATE, Action.RIGHT, -1, None, True)

        agent.learn(*first)
        self.assertEqual(list(agent.memory), [first])

        agent.learn(*second)
        self.assertEqual(list(agent.memory), [first, second])

        agent.learn(*third)
        self.assertEqual(list(agent.memory), [second, third])

    def test_generated_seed_can_recreate_initial_weights(self):
        agent = DQNAgent()
        recreated = DQNAgent(seed=agent.seed)
        for actual, expected in zip(agent.model.parameters(), recreated.model.parameters()):
            self.assertTrue(torch.equal(actual, expected))

    def test_encoding_shape_and_action_order(self):
        agent = DQNAgent(seed=1, epsilon=0)
        for direction in range(4):
            state = (True, False, True, direction, True, False, False, True)
            expected = (1, 0, 1) + tuple(int(direction == i) for i in range(4)) + (1, 0, 0, 1)
            self.assertEqual(agent.encode_state(state), expected)
        inputs = torch.tensor([agent.encode_state(STATE)], dtype=torch.float32)
        self.assertEqual(tuple(agent.model(inputs).shape), (1, 3))

        # Force each output to be the unique best action, independent of the input.
        with torch.no_grad():
            for parameter in agent.model.parameters():
                parameter.zero_()
            for index, action in enumerate((Action.LEFT, Action.STRAIGHT, Action.RIGHT)):
                agent.model[-1].bias.zero_()
                agent.model[-1].bias[index] = 2
                self.assertEqual(agent.choose_action(STATE), action)

    def test_learning_targets_and_weight_update(self):
        for done, expected_target in ((False, 2.8), (True, 1.0)):
            with self.subTest(done=done):
                agent = DQNAgent(seed=1, gamma=0.9, batch_size=1)
                with torch.no_grad():
                    for parameter in agent.model.parameters():
                        parameter.zero_()
                    agent.model[-1].bias[1] = 2.0
                before = [parameter.detach().clone() for parameter in agent.model.parameters()]
                with patch("torch.nn.functional.smooth_l1_loss", wraps=torch.nn.functional.smooth_l1_loss) as loss:
                    agent.learn(STATE, Action.STRAIGHT, 1, None if done else NEXT_STATE, done)
                prediction, target = loss.call_args.args
                self.assertAlmostEqual(prediction.item(), 2.0)
                self.assertAlmostEqual(target.item(), expected_target, places=6)
                self.assertTrue(prediction.requires_grad)
                self.assertFalse(target.requires_grad)
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, agent.model.parameters())))
                self.assertTrue(all(torch.isfinite(parameter).all() for parameter in agent.model.parameters()))

    def test_evaluation_does_not_learn_or_decay(self):
        agent = DQNAgent(seed=1, epsilon=0)
        before = [parameter.detach().clone() for parameter in agent.model.parameters()]
        with patch.object(agent, "learn", wraps=agent.learn) as learn, patch.object(agent, "decay", wraps=agent.decay) as decay:
            with redirect_stdout(StringIO()):
                results = run_episodes(size=3, runs=2, agent=agent, game_seed=2, train=False)
        self.assertEqual(len(results), 2)
        learn.assert_not_called()
        decay.assert_not_called()
        for old, new in zip(before, agent.model.parameters()):
            self.assertTrue(torch.equal(old, new))

class ResultsTest(unittest.TestCase):
    def test_progress_supports_redirected_windows_output(self):
        output = StringIO()
        with redirect_stdout(output):
            run_episodes(size=3, runs=1, agent=RandomAgent(1), game_seed=1, progress_bar=True)
        output.getvalue().encode("cp1252")

    def test_training_reports_epsilon_used_and_rolling_scores(self):
        agent = QLearningAgent(seed=1, epsilon=1, epsilon_decay=0.5)
        with redirect_stdout(StringIO()):
            results = run_episodes(size=3, runs=3, agent=agent, game_seed=1, train=True)
        self.assertEqual([row["epsilon"] for row in results], [1, 0.5, 0.25])
        for index, row in enumerate(results):
            self.assertEqual(row["rolling_average"], sum(r["score"] for r in results[:index + 1]) / (index + 1))

    def test_compiles_episode_results(self):
        results = [
            {"episode": 0, "score": 1, "steps": 10, "max_possible_score": 98},
            {"episode": 1, "score": 3, "steps": 8, "max_possible_score": 98},
            {"episode": 2, "score": 2, "steps": 10, "max_possible_score": 98},
        ]

        self.assertEqual(compile_results(results), {
            "average_score": 2,
            "median_score": 2,
            "best_score": 3,
            "best_score_episode": 1,
            "average_survival": 28 / 3,
            "best_survival": 10,
            "best_survival_episode": 0,
            "max_possible_score": 98,
        })
        self.assertEqual(compile_results(results[:2])["median_score"], 2)

    def test_episode_results_require_verbose_level_one(self):
        def finish_episode(_ep, _runs, game, _agent, _verbose, _train, _trace):
            game.score = 2
            game.steps = 5
            return {"truncated": True}

        with patch("agent.run_episode", side_effect=finish_episode):
            quiet = StringIO()
            with redirect_stdout(quiet):
                run_episodes(size=3, runs=1, verbose=0, agent=RandomAgent(1), game_seed=1)
            logged = StringIO()
            with redirect_stdout(logged):
                run_episodes(size=3, runs=1, verbose=1, agent=RandomAgent(1), game_seed=1)

        self.assertNotIn("{'episode':", quiet.getvalue())
        self.assertIn("{'episode': 0, 'grid_size': 3, 'max_possible_score': 7, 'score': 2, 'steps': 5, 'truncated': True}", logged.getvalue())

    def test_terminal_reward_moves_prediction_toward_target(self):
        import torch
        agent = DQNAgent(seed=1, epsilon=0, batch_size=1)

        # Arrange: make every initial Q-value zero.
        with torch.no_grad():
            for parameter in agent.model.parameters():
                parameter.zero_()

        encoded_state = agent.encode_state(STATE)
        state_tensor = torch.tensor([encoded_state], dtype=torch.float32)

        with torch.no_grad():
            q_values_before = agent.model(state_tensor)
            prediction_before = q_values_before[0,1].item()

        agent.learn(
            STATE,
            Action.STRAIGHT,
            reward=1,
            next_state=None,
            done=True,
        )

        with torch.no_grad():
            q_values_after = agent.model(state_tensor)
            prediction_after = q_values_after[0,1].item()

        self.assertLess(abs(1.0 - prediction_after), abs(1.0 - prediction_before),)

    @unittest.skip("Unfinished learning exercise; replace the placeholder assertion before enabling")
    def test_nonterminal_update_uses_future_reward(self):
        import torch
        agent = DQNAgent(seed=1, epsilon=0, gamma=0.9)

        # Arrange: make every initial Q-value zero.
        with torch.no_grad():
            for parameter in agent.model.parameters():
                parameter.zero_()
                agent.model[-1].bias[2] = 2.0

        encoded_state = agent.encode_state(STATE)
        state_tensor = torch.tensor([encoded_state], dtype=torch.float32)

        with torch.no_grad():
            q_values_before = agent.model(state_tensor)
            prediction_before = q_values_before[0,1].item()

        print(q_values_before)

        agent.learn(
            STATE,
            Action.STRAIGHT,
            reward=1,
            next_state=None,
            done=True,
        )

        self.assertTrue(1)

    
if __name__ == "__main__":
    unittest.main()
