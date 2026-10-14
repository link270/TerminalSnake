# Recording and checkpoints

The recorder compiles data only. It does not generate visuals, interpretations, or
narration. Existing agents keep their learning algorithms; DQN internals are not
recorded yet. PyTorch is needed only when using a DQN agent or its checkpoints.

## Quick examples

Train with summaries and initial/final checkpoints, without step recording:

```console
python agent.py --agent q_learning --agent_seed 10 --game_seed 50 -tr 4000 -er 1000 --save-checkpoints --output-dir Output/q_first
```

Train 100,000 games, record five spaced-out games per phase in detail, and save a
checkpoint every 10,000 training games (plus initial and final checkpoints):

```console
python agent.py -tr 100000 -er 1000 --record detailed --record-episodes 5 --record-every 1000 --checkpoint-every 10000 --output-dir Output/q_long
```

The selected training episode IDs above are 0, 1000, 2000, 3000, and 4000. Only
evaluation episode 0 is selected because the evaluation phase has 1,000 episodes.
Sampling and the limit restart independently for each phase and each invocation.

Record all experience in a short random baseline:

```console
python agent.py --agent random -er 10 --record experience --record-episodes 0 --output-dir Output/random_sample
```

Evaluate a saved agent on a different environment seed, without training:

```console
python agent.py --load-checkpoint Output/q_first/checkpoints/episode_00004000.json --eval-seed 123 --eval-agent-seed 0 -er 20 --record detailed --record-episodes 0 --output-dir Output/q_eval_123
```

Resume for 2,000 additional training episodes:

```console
python agent.py --load-checkpoint Output/q_first/checkpoints/episode_00004000.json --resume-training -tr 2000 -er 0 --save-checkpoints --output-dir Output/q_resumed
```

That run starts with episode ID 4000 and saves its final checkpoint after 6000
completed episodes. Loading without `--resume-training` defaults to evaluation
only. An explicit positive `-tr` with a loaded checkpoint requires that flag.
The saved agent type and board size are inferred. Resume uses the saved size and
environment seed; incompatible overrides are rejected.

## Recording controls

| Option | Behavior |
| --- | --- |
| `--record none` | Default: metadata and episode CSVs only |
| `--record experience` | Also record selected complete episodes, including board states and transitions |
| `--record detailed` | Also record Q-learning decisions/updates and random-agent sampling details |
| `--record-episodes N` | At most N recorded episodes per phase; default 5; 0 means unlimited |
| `--record-every N` | Sample every Nth episode starting with the first; default 1 |
| `--save-checkpoints` | Save initial and final training state |
| `--checkpoint-every N` | Also save at multiples of N completed training episodes; implies checkpoint saving |
| `--output-dir PATH` | Use a new explicit directory; existing directories are rejected |

Recordings stream to disk one transition at a time. Disabled or unselected episodes
do not construct board snapshots or detailed decision/update dictionaries.
Checkpoints are independent of the recording level and are disabled by default.
Saving checkpoints requires a positive training episode count. Random agents can
also use `-tr` for checkpointable episodes, although they do not learn.

## Output layout

```text
Output/<UTC timestamp>_<agent>_<unique suffix>/
  run.json
  training.csv
  evaluation.csv
  experience/
    training/episode_00000000.jsonl
    evaluation/episode_00000000.jsonl
  checkpoints/
    episode_00000000.json
    episode_00004000.json
```

Only requested phases and artifacts appear. DQN checkpoints use `.pt` rather than
`.json`. Earlier flat CSV files are left untouched. Filenames identify episode
IDs for traces and completed episode counts for checkpoints; those differ by one.

`run.json` includes a unique run ID, schema version, effective configuration,
training/evaluation seeds, checkpoint lineage, recording settings, source-file
hashes, Python version, action/feature labels, coordinates, rewards, and limits.
For DQN it also records PyTorch version and layer sizes.

## Episode summaries

CSVs retain episode, board size, score, steps, truncation, epsilon, Q-table size,
and the training rolling average where applicable. New fields are total reward,
true environment termination, ending reason, and `epsilon_used` for that episode.
The existing `epsilon` field is measured after episode-end decay.

Endings distinguish wall collision, body collision, board completion, and the
starvation cutoff. The current learner suppresses bootstrapping for both true
termination and starvation truncation; the records explicitly preserve that
behavior. Total reward is distinct from food score (a collision costs one reward).

Rolling averages use up to 100 episodes in the current invocation. They restart
when resuming; combine the CSVs by episode ID to calculate a continuous curve.
Q-table entry count is not a state-visit count. Visits and state-action frequencies
can be computed from captured transitions, but sampled recordings do not establish
whole-run visitation counts.

## Experience JSONL, schema version 1

Each line is a JSON object. Every record includes `run_id`, `phase`, `episode`,
`schema_version`, and `event`. Each episode contains:

1. `episode_start`: initial board, raw state, and epsilon.
2. `transition`: step number; boards before/after; state and next state; selected
   action; attempted head position; reward; termination/truncation flags; ending
   reason; `done_for_learning`; optional `decision` and `update` objects.
3. `episode_end`: outcome, total reward, steps, score, and epsilon after decay.

`(run_id, phase, episode, step)` identifies a transition. Episode IDs start at zero;
transition steps start at one. Board coordinates are `[x, y]`, with the snake head
first. Direction IDs are right/down/left/up = 0/1/2/3. The default raw state has eight
fields; their order is specified in `run.json`. With `--diagonal-obstacles`, Q-learning
and random recordings append `obstacle_up_left`, `obstacle_up_right`,
`obstacle_down_right`, `obstacle_down_left` (absolute board directions). These use
the existing collision predicate, including the exclusion of the tail.
`settings.diagonal_obstacles` identifies the mode; absent means the original mode.
DQN expands direction into four
one-hot features internally; encoded tensors/activations are not recorded yet.

The final board is available even when `next_state` is null after termination.
On collisions, the attempted head explains the move that failed while the stored
snake remains at its previous position. Starvation retains a valid next state,
but `done_for_learning` is true. A missing `episode_end` indicates an interrupted
recording and must not be interpreted as a completed game.

In detailed mode, Q-learning decisions contain the three pre-update Q-values in
left/straight/right order, tied best actions, epsilon, whether the state was
already known, and `method` (`exploration` or `greedy`). Updates contain old/new
chosen-action values, alpha, gamma, target, TD error, next-state values when used,
and the actual bootstrap value and suppression flag. Evaluation has no updates.
Random decisions contain `method: random` and uniform action probabilities.
DQN gets common experience records only; its `decision` and `update` are null.

## Checkpoints and evaluation

One checkpoint supports both inference and resuming. It contains configuration,
the Q-table or DQN weights, agent RNG state, completed episode count, board size,
environment seed, and environment RNG state. DQN also stores its Adam optimizer
and CPU PyTorch RNG state. Current DQN execution is CPU-only. No replay buffer or
target network is added by this change.

Checkpoints also save `environment.diagonal_obstacles`. Loading automatically
restores this mode for evaluation and resuming. Explicit mode overrides must match;
old checkpoints without this field use standard observations. Q-tables from the
two modes are not interchangeable.

Checkpoints are captured at episode boundaries, after learning and epsilon decay;
the episode-zero checkpoint is captured before the first played episode. Resume
restores randomness before the next reset. Matching execution environments are
needed for exact reproducibility, especially for PyTorch. Experience logs are
for inspection/replay; this change does not add offline training from those logs.

CLI evaluation disables learning and sets epsilon to zero. It defaults to
`game_seed + 1` for the environment and `--eval-agent-seed 0` for actions/ties,
then restores the agent's training epsilon/RNG. It does not add unseen states to
the Q-table. Reuse a checkpoint and vary `--eval-seed` for further evaluations.
Each phase currently uses one continuous environment RNG stream, not individually
seeded episodes. Different policies can consume that stream differently, so a
shared seed alone does not mean they encounter identical food layouts throughout.
