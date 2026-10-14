# Terminal Snake

Play with the arrow keys (or `Q` to quit):

```console
python snake.py
```

An agent can run games without terminal rendering:

```python
from snake import Action, SnakeGame

game = SnakeGame(render=False)
observation = game.reset()
while not game.done:
    observation, reward, done = game.step(Action.STRAIGHT)
```

Actions are `-1` (turn left), `0` (continue straight), and `1` (turn right).
The observation is `(snake_positions, food_position)`; rewards are `1` for food,
`-1` for losing, and `0` otherwise.

Train an agent and save reusable checkpoints:

```console
python agent.py --agent q_learning -tr 4000 -er 1000 --save-checkpoints
```

Each command writes to a new folder under `Output/`. Extra experience recording is
off by default. Add `--record detailed --record-episodes 5` to capture five games
per phase, including Q-learning decisions and updates.

See [Recording and checkpoints](Docs/Recording.md) for recording levels, the data
format, evaluation with different seeds, and resuming training. Random and Q-table
agents use the Python standard library; the DQN agent also requires PyTorch.
