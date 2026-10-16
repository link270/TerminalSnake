# Terminal Snake

## Setup and tests

Verified on Windows with Python 3.14.5 and CPU PyTorch 2.14.0. The code requires
Python 3.12 or newer; dependency wheels must also support your Python version.
Use the existing virtual environment, or create one with `python -m venv .venv`.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -B -m unittest -v
```

The examples below use `python` from an activated environment. You can instead
use `.\.venv\Scripts\python.exe` without activating it.

## Play

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

Pass `Action.LEFT`, `Action.STRAIGHT`, or `Action.RIGHT` to `step()`.
Their underlying values are `-1`, `0`, and `1`; raw integers are not accepted.
The observation is `(snake_positions, food_position)`; rewards are `1` for food,
`-1` for losing, and `0` otherwise.

## Recording, checkpoints, and observations

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

## Run an experiment

```powershell
python agent.py --agent dqn --size 10 --agent_seed 1 --game_seed 100 --eval_game_seed 200 -tr 1000 -er 100 --plot
python agent.py --agent q_learning --size 10 --agent_seed 1 --game_seed 100 --eval_game_seed 200 -tr 1000 -er 100 --plot
python agent.py --agent random --size 10 --agent_seed 1 --game_seed 100 --eval_game_seed 200 -er 100 --plot
```

These are starting comparisons, not a sufficient training budget or evidence
that DQN should outperform tabular learning. DQN has replay storage and sampling,
but still updates from the latest single move after warm-up. Using the sampled
batch for predictions/targets, then adding a target network, are the next steps.

Each invocation prints a unique folder under `Output/` containing:

- `run.json`: resolved seeds, initial agent configuration, architecture, game rules,
  episode counts, source hashes, and Python/PyTorch versions. This replaces the
  local runner's `settings.json`; older output folders remain readable by plotting.
- `training.csv` and/or `evaluation.csv`: one row per episode for phases that ran.
- `summary.json`: per-phase scores, survival lengths, and starvation-cutoff counts.
- `results.png`: generated when `--plot` is supplied.

Repeated runs get separate folders, even with identical seeds. `--output-dir`
selects a new exact run directory; the older `--output_dir` option selects a
parent for an automatically named run folder. They cannot be combined.
`--eval-seed` and `--eval_game_seed` are aliases. Training and evaluation use
distinct game seeds; evaluation defaults to the training seed plus one.
Evaluation has its own action RNG seed (`--eval-agent-seed`, default 0), then
restores the training RNG and epsilon. It disables exploration and learning;
greedy ties are still broken randomly.

Use `-er 0` for training only, or `-tr 0` for evaluation only. Evaluation-only
creates a fresh, untrained agent unless `--load-checkpoint` is supplied.
The random agent defaults to evaluation only; an explicit `-tr` runs checkpointable
episodes without learning. No files are written if no episodes are requested.

The no-food cap is `2 * size * size` moves. It currently counts as a terminal
task outcome for learning, with zero future value and the move's existing reward
(normally zero). `truncated` identifies these episodes in CSVs. This deliberately
preserves the existing task definition; treating it as a time limit with future
value is a separate experiment.

The epsilon column records the value used **during** each episode, before its
end-of-episode decay, matching `epsilon_used`. `epsilon_after` records the decayed
value. Older exports may record the post-decay value as `epsilon`; plotting prefers
`epsilon_used` when present. A rolling average uses up to the latest 100 training
scores in this invocation, including the current episode; it restarts on resume.

## Plot saved results

No training is needed to redraw a plot:

```powershell
python plot_results.py "Output/<experiment-folder>"
```

It writes `results.png` in that folder. Training score and rolling average share
the first panel; epsilon has its own panel. Evaluation is a separate score
histogram with mean, median, and best score, so exploratory training performance
is not mixed into evaluation. Training-only and evaluation-only runs are supported.

Older CSV exports work too:

```powershell
python plot_results.py --training "Output/training_results_10000_6736012832368093329_q_learning.csv" --evaluation "Output/evaluation_results_10000_6736012832368093329_q_learning.csv" --output "Output/legacy_q_learning.png" --title "Earlier tabular Q-learning run"
```

Either CSV may be omitted. Training CSVs need `episode`, `score`, and `rolling_average`;
the exploration panel uses `epsilon_used` or `epsilon` if present. Random-agent
training has no epsilon curve. Evaluation CSVs need `score`. Matplotlib saves the PNG without
opening a window. The plotting command does not import the agents or PyTorch.

### Baseline before experience replay

Recorded October 2, 2026 with the pre-recording, pre-replay runner: 10 x 10 board,
agent seed 1, training game seed 100, evaluation game seed 200, 1,000 training
episodes for learning agents, and 100 evaluation episodes each.

| Agent | Mean evaluation score | Median | Best | Mean survival (steps) | Starvation cutoffs |
| --- | ---: | ---: | ---: | ---: | ---: |
| Random | 0.15 | 0 | 1 | 19.84 | 0 |
| Tabular Q-learning | 13.80 | 13 | 31 | 122.41 | 2 |
| DQN, single-transition updates | 18.99 | 18 | 31 | 149.34 | 0 |

Local CSVs, settings, summaries, and charts are under `Output/baseline_step4/`.
These generated files are ignored by Git. This is one seed and one training
budget, not evidence of a general ranking. The merged runner independently seeds
evaluation actions, and DQN now has replay warm-up/sampling, so the current commands
do not reproduce these historical numbers exactly. Establish a fresh baseline
before comparing future changes, then repeat across seeds.

## Learning checkpoint

The current checks cover DQN seed initialization (including generated seeds),
state encoding, output shape/action order, numerical terminal/nonterminal targets,
weight updates, and evaluation without learning. Runner checks cover phase skips,
seeds, independent outputs, metadata, and plots.

Replay retention, capacity, warm-up, and sampling are in place in `DQNAgent`.
Resume by replacing the single-transition prediction/target calculations with
batch calculations. Preserve the runner's optional `update=None` recording argument.
Single-update tests use `batch_size=1`; adapt them when implementing batched targets.

The separate nonterminal learning exercise brought from the other computer still
has a placeholder assertion and is explicitly skipped until you finish it. Its
numerical target is already covered by the local DQN test. See
[Learning handoff](Docs/Learning_Handoff.md) for the teaching context.
