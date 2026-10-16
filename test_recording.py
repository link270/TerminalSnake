import importlib.util
import json
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from agent import run_episode, run_episodes
from agent_cli import main
from agent_defs import DQNAgent, QLearningAgent, RandomAgent
from checkpoints import load_checkpoint, save_checkpoint
from recording import RunOutput
from snake import Action, SnakeGame


def quiet_run(**kwargs):
    with redirect_stdout(StringIO()):
        return run_episodes(**kwargs)


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class RecordingTest(unittest.TestCase):
    def test_recording_preserves_results_and_randomness(self):
        for agent_type in (RandomAgent, QLearningAgent):
            with self.subTest(agent=agent_type.__name__), TemporaryDirectory() as temp:
                plain, recorded = agent_type(seed=7), agent_type(seed=7)
                output = RunOutput(Path(temp) / "run", recorded, {}, "detailed", 0)
                options = dict(size=4, runs=12, game_seed=8, train=True)
                expected = quiet_run(agent=plain, **options)
                actual = quiet_run(agent=recorded, output=output, **options)
                self.assertEqual(actual, expected)
                self.assertEqual(recorded.q_table, plain.q_table)
                self.assertEqual(recorded.rng.getstate(), plain.rng.getstate())
                events = read_events(output.directory / "experience/training/episode_00000000.jsonl")
                transitions = events[1:-1]
                self.assertEqual(events[0]["event"], "episode_start")
                self.assertEqual(events[-1]["event"], "episode_end")
                self.assertEqual(sum(t["reward"] for t in transitions), actual[0]["total_reward"])
                self.assertEqual(len(transitions), actual[0]["steps"])
                for previous, current in zip(transitions, transitions[1:]):
                    self.assertEqual(previous["after"], current["before"])
                for transition in transitions:
                    self.assertIn("attempted_head", transition)
                    if agent_type is QLearningAgent:
                        update = transition["update"]
                        self.assertAlmostEqual(update["new_q"], update["old_q"] + update["alpha"] * update["td_error"])
                        self.assertAlmostEqual(update["target"], transition["reward"] + update["gamma"] * update["bootstrap_value"])
                        self.assertEqual(transition["decision"]["q_values"][transition["action"] + 1], update["old_q"])
                    else:
                        self.assertEqual(transition["decision"]["method"], "random")
                        self.assertIsNone(transition["update"])

    def test_recording_limits_and_disabled_mode(self):
        with TemporaryDirectory() as temp:
            for level in ("none", "experience", "detailed"):
                agent = QLearningAgent(seed=4)
                output = RunOutput(Path(temp) / level, agent, {}, level, record_episodes=2, record_every=3)
                quiet_run(size=3, runs=8, game_seed=5, agent=agent, output=output)
                files = sorted(output.directory.glob("experience/evaluation/*.jsonl"))
                self.assertEqual([p.stem for p in files], [] if level == "none" else ["episode_00000000", "episode_00000003"])
                if files:
                    transition = read_events(files[0])[1]
                    self.assertIsNone(transition["update"])
                    self.assertEqual(transition["decision"] is None, level == "experience")

    def test_truncation_is_distinct_from_death_and_records_actual_bootstrap(self):
        with TemporaryDirectory() as temp:
            agent = QLearningAgent(seed=1)
            game = SnakeGame(size=5, seed=1, render=False)
            original_reset = game.reset

            def reset():
                original_reset()
                game.food = (0, 0)
                return game.observation

            output = RunOutput(Path(temp) / "run", agent, {}, "detailed")
            with output.episode("training", 0, 0) as trace, redirect_stdout(StringIO()):
                with patch.object(game, "reset", side_effect=reset), patch.object(agent, "choose_action", return_value=Action.RIGHT):
                    outcome = run_episode(0, 1, game, agent, 0, True, trace)
            last = read_events(output.directory / "experience/training/episode_00000000.jsonl")[-2]
            self.assertTrue(outcome["truncated"])
            self.assertFalse(outcome["terminated"])
            self.assertEqual(last["termination_reason"], "starvation_limit")
            self.assertIsNotNone(last["next_state"])
            self.assertTrue(last["update"]["bootstrap_suppressed"])

    def test_evaluation_does_not_add_states_to_q_table(self):
        agent = QLearningAgent(seed=1, epsilon=0)
        quiet_run(agent=agent, size=3, runs=4, game_seed=7)
        self.assertEqual(agent.q_table, {})


class CheckpointTest(unittest.TestCase):
    def check_resume(self, agent_type, episodes=12):
        with TemporaryDirectory() as temp:
            config = {"seed": 3, **({"batch_size": 2} if agent_type is DQNAgent else {})}
            continuous = agent_type(**config)
            expected = quiet_run(agent=continuous, size=3, runs=episodes, game_seed=11, train=True)
            split = agent_type(**config)
            first = RunOutput(Path(temp) / "first", split, {}, "detailed")
            quiet_run(agent=split, size=3, runs=episodes // 2, game_seed=11, train=True,
                      output=first, checkpoint_every=0)
            extension = ".pt" if agent_type is DQNAgent else ".json"
            path = first.directory / "checkpoints" / f"episode_{episodes // 2:08d}{extension}"
            restored, resume = load_checkpoint(path)
            if agent_type is DQNAgent:
                event = read_events(first.directory / "experience/training/episode_00000000.jsonl")[1]
                self.assertIsNone(event["decision"])
                self.assertIsNone(event["update"])
            second = RunOutput(Path(temp) / "second", restored, {})
            actual = quiet_run(agent=restored, size=3, runs=episodes // 2, game_seed=11, train=True,
                               output=second, checkpoint_every=0, resume=resume)
            for got, wanted in zip(actual, expected[episodes // 2:]):
                # Rolling averages describe each invocation's local history.
                self.assertEqual({k: v for k, v in got.items() if k != "rolling_average"},
                                 {k: v for k, v in wanted.items() if k != "rolling_average"})
            self.assertEqual(restored.q_table, continuous.q_table)
            self.assertEqual(restored.epsilon, continuous.epsilon)
            self.assertEqual(restored.rng.getstate(), continuous.rng.getstate())
            if agent_type is DQNAgent:
                import torch

                for key, tensor in continuous.model.state_dict().items():
                    self.assertTrue(torch.equal(tensor, restored.model.state_dict()[key]), key)
                self.assertEqual(restored.batch_size, 2)
                self.assertEqual(restored.memory, continuous.memory)
                self.assertEqual(restored.memory.maxlen, continuous.memory.maxlen)
                expected_optimizer = continuous.optimizer.state_dict()["state"]
                actual_optimizer = restored.optimizer.state_dict()["state"]
                self.assertTrue(expected_optimizer, "The resume check must exercise real optimizer updates")
                for parameter, values in expected_optimizer.items():
                    for key, value in values.items():
                        self.assertTrue(torch.equal(value, actual_optimizer[parameter][key]))
            # Episode-zero snapshots can also resume before the first game.
            self.assertTrue((first.directory / "checkpoints" / f"episode_00000000{extension}").exists())

    def test_q_learning_and_random_resume_exactly(self):
        for agent_type in (QLearningAgent, RandomAgent):
            with self.subTest(agent=agent_type.__name__):
                self.check_resume(agent_type)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_dqn_resume_includes_optimizer(self):
        self.check_resume(DQNAgent, episodes=4)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_legacy_dqn_can_evaluate_but_cannot_claim_exact_resume(self):
        import torch

        with TemporaryDirectory() as temp, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(temp)
            agent = DQNAgent(seed=3)
            game = SnakeGame(size=3, seed=11, render=False)
            path = save_checkpoint(root / "saved", agent, game, 0)
            data = torch.load(path, weights_only=True)
            del data["replay_memory"]
            del data["config"]["batch_size"]
            torch.save(data, path)

            evaluated = main(["--load-checkpoint", str(path), "-er", "2", "--plot",
                              "--record", "experience", "--output-dir", str(root / "evaluation")])
            self.assertTrue((evaluated / "results.png").exists())
            self.assertTrue((evaluated / "experience/evaluation/episode_00000000.jsonl").exists())
            with self.assertRaises(SystemExit):
                main(["--load-checkpoint", str(path), "--resume-training", "-tr", "2", "-er", "0",
                      "--output-dir", str(root / "resume")])
            self.assertFalse((root / "resume").exists())
            restored, resume = load_checkpoint(path)
            with self.assertRaisesRegex(ValueError, "predates replay state"):
                quiet_run(agent=restored, size=3, runs=2, game_seed=11, train=True, resume=resume)

    def test_cli_train_load_evaluate_and_resume(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
            root = Path(temp)
            main(["--size", "3", "--agent_seed", "9", "--game_seed", "2", "-tr", "4", "-er", "0",
                  "--record", "detailed", "--checkpoint-every", "2", "--output-dir", str(root / "train")])
            checkpoints = sorted((root / "train/checkpoints").glob("*.json"))
            self.assertEqual(len(checkpoints), 3)
            checkpoint = checkpoints[-1]
            original = checkpoint.read_bytes()
            for seed in (33, 44):
                main(["--load-checkpoint", str(checkpoint), "--eval-seed", str(seed), "-er", "3",
                      "--record", "experience", "--output-dir", str(root / f"eval_{seed}")])
                metadata = json.loads((root / f"eval_{seed}/run.json").read_text())
                self.assertEqual(metadata["settings"]["evaluation_seed"], seed)
                self.assertFalse((root / f"eval_{seed}/training.csv").exists())
            self.assertEqual(checkpoint.read_bytes(), original)
            main(["--load-checkpoint", str(checkpoint), "--resume-training", "-tr", "2", "-er", "0",
                  "--save-checkpoints", "--output-dir", str(root / "resume")])
            self.assertTrue((root / "resume/checkpoints/episode_00000006.json").exists())
            main(["--agent", "random", "-er", "2", "--output-dir", str(root / "random")])
            self.assertFalse((root / "random/experience").exists())
            self.assertFalse((root / "random/checkpoints").exists())
            with self.assertRaises(SystemExit), redirect_stderr(StringIO()):
                main(["--output-dir", str(root / "random"), "-tr", "0", "-er", "1"])


if __name__ == "__main__":
    unittest.main()
