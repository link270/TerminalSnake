import random
from snake import SnakeGame, Action

game = SnakeGame(render=True)

for ep in range(10):
    observation = game.reset()

    while not game.done:
        action = random.choice(list(Action))
        observation, reward, done = game.step(action)

    print(f"Ep {ep}: score={game.score}")