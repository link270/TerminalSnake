# Snake neural network learning handoff

Updated October 15, 2026. Resume with the unfinished nonterminal learning test in `test_agent.py`, then continue toward experience replay and a target network. The broader curriculum is in [Project_Plan.md](Project_Plan.md).

## How to guide this project

The user wants to learn by implementing the work themselves. Act as a tutor: explain difficult concepts in simple language, give one small step at a time, and let the user try it before continuing. Small snippets are helpful; do not provide whole implementations or complete test solutions unless requested. Review their work when asked, explain what an assertion actually proves, and correct misunderstandings gently. Do not modify code, install dependencies, commit, or push without authorization. Creating this handoff was explicitly requested.

Avoid turning the learning project into an extensive testing exercise. Finish a few useful checks, then advance to the next ML concept. Keep changes small and experiments controlled.

## Current implementation

- Human-playable Snake, a random agent, tabular Q-learning, training/evaluation separation, recording, and checkpoints are implemented.
- Q-table observation comparisons have been run through 50,000 training episodes, with standard and diagonal observations across three seeds. Their summaries are under `Output/` on the original computer.
- `DQNAgent` in `agent_defs.py` is an initial neural Q-learning implementation. It expands eight state fields into 11 inputs by encoding direction as four flags. Its network is 11 → 64 → 64 → 3, with ReLU hidden activations.
- Outputs are estimated Q-values for LEFT, STRAIGHT, RIGHT, in that order. These are expected discounted returns, not action probabilities.
- Each move immediately triggers one learning update using Smooth L1 loss and Adam. The same network supplies the current prediction and the next-state estimate, with gradient tracking disabled for the target.
- There is no replay buffer, minibatch learning, or separate target network yet. DQN does not support the optional diagonal observation mode.
- DQN checkpoint/resume coverage already exists in `test_recording.py`, including optimizer state. Detailed DQN decision/update internals are not recorded yet.
- Older files named `training_results_4000_50_dnq.csv` and `evaluation_results_4000_50_dnq.csv` were checked and are byte-for-byte identical to their Q-learning counterparts. They do not establish DQN performance; run a fresh baseline later.

## Concepts already covered

The user correctly calculated a nonterminal target of `1 + 0.9 * 2 = 2.8`, and a terminal target of `1` for the same immediate reward.

- A tensor is a container of numbers. Input shape `[1, 11]` means one situation with 11 features; `[2, 11]` means two situations.
- A forward pass calls the network to calculate predictions. For one situation its output shape is `[1, 3]`.
- `[0, 1]` selects the first situation's STRAIGHT prediction; `.item()` extracts a Python number.
- `torch.no_grad()` is appropriate for inspecting predictions and constructing targets. The actual learning call needs gradient tracking.
- A previous output does not change automatically after a weight update; call the model again.
- One optimizer step should move a prediction toward the target, not necessarily reach it.
- Checking distance to the target is more accurate than merely checking an increase: moving from 0 to 3 increases the prediction but moves farther from target 1.
- Zeroing all parameters is a controlled test fixture, not a good general training initialization. In these tests the output bias can change even with zero hidden weights.

## Completed terminal test

`ResultsTest.test_terminal_reward_moves_prediction_toward_target` in `test_agent.py`:

1. Creates a DQN agent and zeros all network parameters.
2. Encodes `STATE`, constructs a float32 tensor, and captures the STRAIGHT prediction.
3. Learns from reward `1`, `next_state=None`, and `done=True`.
4. Calculates a fresh prediction and asserts that its absolute distance from `1.0` decreased.

The earlier version, which asserted only that the prediction increased, passed when run in `.venv`. The user then replaced that assertion with the distance comparison; that final version has been inspected but not rerun during this handoff. The test currently lives in `ResultsTest`; moving it to a DQN-specific class is optional housekeeping, not the next learning objective.

## Exact place to resume

`ResultsTest.test_nonterminal_update_uses_future_reward` is unfinished. The user created an agent with gamma `0.9`, zeroed its parameters, and set the final layer's RIGHT bias to `2.0`. They confirmed output `[0, 0, 2]`.

At the time of this handoff, the file still contains:

- The RIGHT bias assignment inside the parameter-zeroing loop. It ends with the intended values, but should be moved just after the loop, still inside `no_grad()`, for clarity.
- A captured STRAIGHT prediction for `STATE` and a temporary print.
- The copied terminal learning call: reward `1`, `next_state=None`, `done=True`.
- A placeholder `self.assertTrue(1)`, which always passes and checks nothing about learning.

The next exercise has already been explained, but the user has not yet completed it in the file:

1. Use STRAIGHT with reward `0`, `next_state=NEXT_STATE`, and `done=False`.
2. Reason about the target: `0 + 0.9 * 2 = 1.8`.
3. After learning, recalculate the STRAIGHT prediction for the original `STATE`.
4. Replace the placeholder with a comparison showing that its distance from `1.8` decreased.

The fixture predicts `[0, 0, 2]` for every state because all weights are zero and only the final bias is nonzero. STRAIGHT starts at zero; RIGHT supplies the best next-state value. An upward movement of STRAIGHT with zero immediate reward demonstrates the influence of future reward.

This behavioral assertion does not establish that the internal target was exactly `1.8`: other positive targets could cause a similar small update. If useful, follow it with a short guided exercise observing the target passed into the loss function, using a wrapping mock that preserves the real calculation. Compare terminal and nonterminal cases with the same nonzero next-state values. Explain why inspecting the target is different from inspecting the optimizer's output; avoid a large mocking detour.

## Remaining learning sequence

1. **Finish the current tests.** Review the user's attempt and run only the relevant tests initially. Explain that terminal transitions stop future-value contributions. After the small test group is complete, run the existing suite once; don't keep broadening coverage without a reason.
2. **Measure the current neural agent.** Run a fresh, explicitly selected `--agent dqn` baseline. Compare random, Q-table, and DQN agents on the same board size and standard observations, using common evaluation settings and several training seeds. Save checkpoints and manifests. Start with a small smoke run before choosing a larger training budget. Use mean/median score, training curves, and starvation outcomes; one good game or lower loss alone is not proof of a stronger policy. A neural network need not outperform the table on this compact state representation.
3. **Implement experience replay together.** Explain a stored transition `(state, action, reward, next_state, done)` as one remembered experience. Let the user add bounded storage, random sampling, then minibatches in separate steps. Explain how shuffling reduces the dominance of consecutive, closely related moves. Walk through batch shapes and selecting each sample's chosen action value. Handle terminal next states deliberately.
4. **Implement a target network together.** Explain that a network learning from its own changing predictions has a moving target. Keep a separate network for future-value estimates and periodically copy the learning network's weights into it. Teach gradient isolation and a clearly defined update interval. Compare against the replay-only baseline before changing other settings.
5. **Keep persistence correct as features change.** Replay memory, sampling randomness, target-network weights, and update counters may affect exact training continuation. Revisit checkpoint/resume behavior when those features are added; don't silently preserve an exact-resume claim while omitting new state.
6. **Analyze learning and run controlled experiments.** Build on existing CSVs and recording. Explain epsilon, discount factor, optimizer learning rate, loss versus game score, noisy curves, and variation across seeds. Change one factor at a time: exploration, gamma, learning rate, batch size, replay capacity, network size, or target-copy frequency. The current CLI does not expose all these settings; inspect available options before suggesting commands.
7. **Explore richer observations after the basic DQN is understood.** Consider a local grid or full board. Explain what handcrafted features hide and why identical compact observations can correspond to different traps. Update input shapes and comparisons deliberately. Treat reward shaping as a separate experiment, looking for unintended incentives.

Use the original project plan as a curriculum, not a requirement to implement every possible experiment. The learning milestone is being able to explain and measure each change.

## Continue on another computer

`requirements.txt` currently contains `torch` without a version pin. A local `.venv` exists on the original computer and PyTorch is installed there. Recreate the environment after pulling the repo; `.venv` is ignored by Git. From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__)"
.\.venv\Scripts\python.exe -m unittest test_agent.ResultsTest.test_terminal_reward_moves_prediction_toward_target -v
```

Select `.venv\Scripts\python.exe` in the editor. On macOS/Linux, the environment executable is `.venv/bin/python`. Tests use built-in `unittest`; pytest is not required. The earlier passing test emitted a missing-NumPy warning, but NumPy was not required for that test's tensor operations. Pinning the working PyTorch version can be a later reproducibility step.

After completing the nonterminal test, run it by its full unittest name. Run the broader suite with `python -m unittest discover -v` using the environment's Python. A passing placeholder assertion is not validation of the unfinished test.

`Output/` is ignored by Git, so existing training data and checkpoints will not accompany a normal push. Copy selected artifacts separately if needed. Include this handoff along with the current tests and requirements when transferring the project. The assistant did not commit or push anything for this handoff.

Suggested prompt for the next chat: “Read Docs/Learning_Handoff.md and the current tests. Continue tutoring me one small step at a time from the unfinished nonterminal DQN test. Explain difficult concepts simply, and let me write the code.”
