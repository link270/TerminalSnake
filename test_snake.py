import unittest

from snake import SnakeGame


class SnakeGameTest(unittest.TestCase):
    def test_step_turn_eat_and_collision(self):
        game = SnakeGame(render=False)
        game.food = (5, 4)
        observation, reward, done = game.step(-1)
        self.assertEqual((observation, reward, done), ((((5, 4), (5, 5), (4, 5)), game.food), 1, False))

        game.snake = [(9, 5), (8, 5)]
        game.direction = 0
        _, reward, done = game.step(0)
        self.assertEqual((reward, done), (-1, True))

    def test_rejects_invalid_action(self):
        with self.assertRaises(ValueError):
            SnakeGame(render=False).step(2)


if __name__ == "__main__":
    unittest.main()
