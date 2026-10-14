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

Optional observation experiment (Q-learning and random agents only):

```console
python agent.py --diagonal-obstacles --size 6 -tr 5000 -er 300 --save-checkpoints
python experiments/compare_observations.py --output-dir Output/diagonal-comparison
```

The default state remains the original eight features. `--diagonal-obstacles`
appends four boolean obstacle checks in absolute upper-left, upper-right,
lower-right, lower-left order. These reuse the existing collision predicate
(walls or body excluding the tail). Nothing changes in movement, food placement,
rewards, termination, action selection, or learning updates. The extra inputs
are local checks, not predictions of future collisions.

Recordings identify all state fields. Checkpoints save the observation mode;
loading or resuming automatically restores it. Explicitly requesting a different
mode is rejected. Old checkpoints remain standard mode. `--no-diagonal-obstacles`
explicitly selects the original mode for a new agent.

The comparison script runs three matched training seed pairs, records every 50th
game, saves checkpoints, and writes `summary.csv`, `experiment.json`, and per-run
logs. Evaluation uses frozen policies and common seeds; different trajectories
can consume the environment RNG differently, so episode IDs are not matched boards.

Extend those same policies to 20,000 total training games (15,000 additional):

```console
python experiments/compare_observations.py --resume-from Output/diagonal-comparison --training-runs 20000 --output-dir Output/diagonal-comparison-20000
```

The saved exploration rate and RNG states continue unchanged. The new training
CSVs contain only the additional games; evaluation still measures the full
training history. The extension samples detailed recordings every 250 games.
