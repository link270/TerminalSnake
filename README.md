# Terminal Snake

Play with the arrow keys (or `Q` to quit):

```console
python snake.py
```

An agent can run games without terminal rendering:

```python
from snake import SnakeGame

game = SnakeGame(render=False)
observation = game.reset()
while not game.done:
    observation, reward, done = game.step(0)
```

Actions are `-1` (turn left), `0` (continue straight), and `1` (turn right).
The observation is `(snake_positions, food_position)`; rewards are `1` for food,
`-1` for losing, and `0` otherwise.
