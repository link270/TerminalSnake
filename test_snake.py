import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from snake import Action, SnakeGame, StepResult, play


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

    @patch("snake._read_key", return_value="q")
    @patch.object(SnakeGame, "render")
    def test_play_uses_frame_delay(self, render, _read_key):
        play(delay=0.15)
        render.assert_called_once_with(0.15)


if __name__ == "__main__":
    unittest.main()
