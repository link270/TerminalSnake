import random
from snake import SnakeGame

game = SnakeGame(render=true)

for ep in range(10):
    observation = game.reset()

    while not game.done:
        action = random.choice()
        observation, reward, done = game.step(action)