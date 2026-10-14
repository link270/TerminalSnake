"""The optional inputs must not change the environment or learning rules."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from agent_cli import main
from agent_defs import QLearningAgent, RandomAgent
from checkpoints import load_checkpoint
from recording import RunOutput
from snake import Action, SnakeGame
from test_recording import quiet_run


class ObservationTest(unittest.TestCase):
    def test_diagonals_are_absolute_and_preserve_existing_tail_semantics(self):
        game = SnakeGame(size=5, render=False, seed=1, diagonal_obstacles=True)
        game.snake = [(2, 2), (2, 1), (1, 1), (1, 2), (1, 3)]
        for heading in range(4):
            game.direction = heading
            self.assertEqual(game.get_state()[8:], (True, False, False, False))
        game.snake = [(0, 0), (0, 1)]
        self.assertEqual(game.get_state()[8:], (True, True, False, True))

    def test_same_actions_produce_identical_games_and_rng(self):
        base = SnakeGame(size=6, seed=8, render=False)
        diagonal = SnakeGame(size=6, seed=8, render=False, diagonal_obstacles=True)
        for _ in range(10):
            self.assertEqual(base.reset(), diagonal.reset())
            for action in [Action.LEFT, Action.STRAIGHT, Action.RIGHT] * 20:
                self.assertEqual(base.get_state(), diagonal.get_state()[:8])
                self.assertEqual(len(base.get_state()), 8)
                self.assertEqual(base.step(action), diagonal.step(action))
                self.assertEqual(base.random.getstate(), diagonal.random.getstate())
                if base.done:
                    break

    def test_random_agent_outcomes_unchanged(self):
        base, diagonal = RandomAgent(seed=7), RandomAgent(seed=7)
        diagonal.diagonal_obstacles = True
        options = dict(size=6, runs=100, game_seed=8)
        self.assertEqual(quiet_run(agent=base, **options), quiet_run(agent=diagonal, **options))

    def test_diagonal_recording_and_resume_are_exact(self):
        with TemporaryDirectory() as temp:
            base, split = QLearningAgent(seed=3), QLearningAgent(seed=3)
            base.diagonal_obstacles = split.diagonal_obstacles = True
            expected = quiet_run(agent=base, size=4, runs=30, game_seed=11, train=True)
            output = RunOutput(Path(temp) / 'first', split, {}, 'detailed', 1)
            quiet_run(agent=split, size=4, runs=15, game_seed=11, train=True,
                      output=output, checkpoint_every=0)
            restored, checkpoint = load_checkpoint(output.directory / 'checkpoints/episode_00000015.json')
            self.assertTrue(restored.diagonal_obstacles)
            actual = quiet_run(agent=restored, size=4, runs=15, game_seed=11, train=True, resume=checkpoint)
            self.assertEqual(restored.q_table, base.q_table)
            self.assertEqual(restored.rng.getstate(), base.rng.getstate())
            self.assertEqual([r['score'] for r in actual], [r['score'] for r in expected[15:]])
            meta = json.loads((output.directory / 'run.json').read_text())
            self.assertEqual(len(meta['state_fields']), 12)
            events = [json.loads(line) for line in
                      (output.directory / 'experience/training/episode_00000000.jsonl').read_text().splitlines()]
            self.assertEqual(len(events[0]['state']), 12)
            self.assertTrue(all(len(row['state']) == 12 for row in checkpoint['q_table']))

    def test_cli_inherits_mode_rejects_mixing_and_loads_legacy(self):
        with TemporaryDirectory() as temp, redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            root = Path(temp)
            for mode in ('standard', 'diagonal'):
                main(['--size', '3', '--agent_seed', '1', '--game_seed', '2', '-tr', '4', '-er', '0',
                      '--save-checkpoints', '--output-dir', str(root / mode)]
                     + (['--diagonal-obstacles'] if mode == 'diagonal' else []))
                path = root / mode / 'checkpoints/episode_00000004.json'
                if mode == 'standard':
                    legacy = json.loads(path.read_text())
                    del legacy['environment']['diagonal_obstacles']
                    path.write_text(json.dumps(legacy))
                main(['--load-checkpoint', str(path), '-er', '2', '--output-dir', str(root / f'{mode}-eval')])
                meta = json.loads((root / f'{mode}-eval/run.json').read_text())
                self.assertEqual(meta['settings']['diagonal_obstacles'], mode == 'diagonal')
                with self.assertRaises(SystemExit):
                    main(['--load-checkpoint', str(path), '--no-diagonal-obstacles' if mode == 'diagonal'
                          else '--diagonal-obstacles', '-er', '1'])
            with self.assertRaises(SystemExit):
                main(['--agent', 'dqn', '--diagonal-obstacles'])


if __name__ == '__main__':
    unittest.main()
