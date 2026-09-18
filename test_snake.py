import unittest
from contextlib import redirect_stdout
from io import StringIO
import os
from unittest.mock import patch

from snake import Action, SnakeGame, StepResult, _read_key, play


class SnakeGameTest(unittest.TestCase):
    def test_step_turn_eat_and_collision(self):
        game = SnakeGame(render=False)
        self.assertEqual(game.steps, 0)
        game.food = (5, 4)
        result = game.step(Action.LEFT)
        self.assertIsInstance(result, StepResult)
        observation, reward, done = result
        self.assertEqual((observation, reward, done), ((((5, 4), (5, 5), (4, 5)), game.food), 1, False))
        self.assertEqual(game.steps, 1)

        game.snake = [(9, 5), (8, 5)]
        game.direction = 0
        _, reward, done = game.step(Action.STRAIGHT)
        self.assertEqual((reward, done), (-1, True))
        self.assertEqual(game.steps, 2)

    def test_rejects_invalid_action(self):
        with self.assertRaises(ValueError):
            SnakeGame(render=False).step(2)

    def test_would_collide_without_changing_game(self):
        game = SnakeGame(size=5, render=False)
        game.snake = [(4, 2), (3, 2)]
        game.direction = 0

        self.assertTrue(game.would_collide(Action.STRAIGHT))
        self.assertFalse(game.would_collide(Action.LEFT))
        self.assertEqual((game.snake, game.direction, game.steps), ([(4, 2), (3, 2)], 0, 0))

        game.snake = [(2, 2), (2, 1), (1, 1), (1, 2)]
        game.direction = 2
        self.assertFalse(game.would_collide(Action.STRAIGHT))

    def test_render_message(self):
        output = StringIO()
        with redirect_stdout(output):
            SnakeGame(render=True).render(message="Agent chose LEFT")
        self.assertTrue(output.getvalue().endswith("Agent chose LEFT\n"))


class TerminalPlayTest(unittest.TestCase):
    @unittest.skipIf(os.name == "nt", "POSIX input test")
    @patch("snake.select.select", return_value=([object()], [], []))
    @patch("snake.sys.stdin", StringIO("\x1b[A"))
    def test_reads_posix_arrow_key(self, _select):
        self.assertEqual(_read_key(), "H")

    @patch("snake._read_key", side_effect=[None, "H", None, "q"])
    @patch("snake.time.sleep")
    @patch.object(SnakeGame, "step")
    @patch.object(SnakeGame, "render")
    def test_play_waits_for_an_arrow_key(self, render, step, sleep, _read_key):
        play(delay=0.15)
        self.assertEqual(render.call_args_list[0].kwargs, {"message": "Press an arrow key to start; Q quits."})
        render.assert_any_call()
        sleep.assert_any_call(0.01)
        sleep.assert_any_call(0.15)
        step.assert_called_once()


if __name__ == "__main__":
    unittest.main()
