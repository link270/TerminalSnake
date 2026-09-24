# Snake Machine Learning Learning Project

## Project Goal

The goal of this project is to learn the fundamentals of machine learning through a small, understandable project rather than relying heavily on prebuilt ML frameworks.

The project uses a simple Snake game as the environment and gradually introduces reinforcement learning concepts.

The emphasis is on understanding:

* how an agent interacts with an environment,
* how game state is represented,
* how rewards influence behavior,
* how exploration and exploitation work,
* how Q-learning updates its knowledge,
* how neural networks can replace explicit lookup tables,
* and how changes to observations and rewards affect learned behavior.

The project should remain small enough that every major part of the learning system can be understood and implemented directly.

---

# Project Progression

## Phase 1 — Human-Playable Snake

### Goal

Build a simple working Snake game before introducing any machine learning.

### Requirements

The game should include:

* a bounded grid,
* a snake represented by grid positions,
* movement in four directions,
* food spawning,
* snake growth when food is eaten,
* collision with walls and the snake's own body,
* a score,
* a game-over state,
* and basic terminal rendering.

### Learning Purpose

This establishes a known working environment before adding ML logic.

The game rules should remain independent from any future learning algorithm.

### Status

Completed.

---

# Phase 2 — Random Agent and Environment Refactor

## Goal

Allow an external agent to control Snake without human input.

The agent itself should have no intelligence yet and should simply choose random actions.

This phase is primarily about converting the Snake game into an environment suitable for reinforcement learning.

## Core Architecture

The game should expose an interface approximately like:

```python
game.reset()
game.step(action)
```

One call to `step()` should advance the simulation exactly one game tick.

The game should not need to know whether the action came from:

* a human,
* a random agent,
* a Q-learning agent,
* or a future neural-network agent.

## Actions

Prefer relative actions:

```text
LEFT
STRAIGHT
RIGHT
```

rather than absolute directions.

The meaning of these actions depends on the snake's current heading.

This reduces the number of possible actions and simplifies future learning.

## Random Agent

Create a basic agent that does nothing except:

```python
random.choice(actions)
```

The random agent acts as the baseline against which future learning agents can be compared.

## Episodes

Introduce the concept of an episode:

```text
Reset game
↓
Play until death / termination
↓
Record result
↓
Reset
```

Run many episodes automatically.

## Statistics

Record basic metrics such as:

* score,
* number of simulation steps,
* average score,
* average episode length,
* and best score.

These establish the random baseline.

## Additional Goals

Separate simulation from rendering so training can run without terminal output or artificial delays.

Add an episode-length or starvation limit so agents cannot remain alive indefinitely without making progress.

Use deterministic random seeds where useful so behavior can be reproduced during debugging.

---

# Phase 3 — Tabular Q-Learning

## Goal

Create the first agent that actually learns from experience.

Use tabular Q-learning rather than a neural network so the learning process remains easy to inspect and understand.

---

## State Representation

Instead of exposing the entire Snake board, start with a simplified state made from boolean values.

Example:

```text
danger_straight
danger_right
danger_left

moving_up
moving_down
moving_left
moving_right

food_up
food_down
food_left
food_right
```

This produces 11 boolean inputs.

There are therefore at most:

```text
2^11 = 2048 states
```

With three possible actions:

```text
2048 × 3 = 6144 possible Q-values
```

This is small enough to store directly in a dictionary.

---

## Q-Table

Each observed state stores an estimated value for every possible action.

For example:

```text
State X:

LEFT        1.5
STRAIGHT    6.8
RIGHT      -4.2
```

These values represent the agent's current estimate of how valuable each action is from that state.

Unseen states should initially have Q-values of zero.

---

## Rewards

Start with a deliberately simple reward system.

Example:

```text
Eat food:       +10
Die:            -10
Normal move:      0
```

Avoid heavy reward shaping initially.

The purpose is to see what behavior emerges from a minimal objective before introducing additional incentives.

---

## Exploration vs Exploitation

Use an epsilon-greedy action policy.

The agent sometimes:

```text
explores → choose a random action
```

and otherwise:

```text
exploits → choose the action with the highest Q-value
```

Start with high exploration and gradually reduce it.

Example:

```text
epsilon starts near 1.0
↓
decays during training
↓
minimum around 0.05
```

Randomly break ties between equally valued actions to avoid introducing unintended action bias.

---

## Q-Learning Update

The central learning equation is:

```text
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:

```text
s   = current state
a   = chosen action
r   = reward
s'  = next state

α   = learning rate
γ   = discount factor
```

Conceptually:

> Move the current estimate slightly toward the immediate reward plus the best expected future reward.

Terminal states should not include future Q-values because no next action exists after the episode ends.

---

## Training Loop

The basic learning process is:

```text
Observe current state
↓
Choose action
↓
Advance game
↓
Receive reward
↓
Observe next state
↓
Update Q-table
↓
Repeat
```

Training should happen without rendering so thousands of episodes can run quickly.

---

## Evaluation

Training and evaluation should be separated.

During evaluation:

```text
epsilon = 0
```

so the agent always follows its learned policy.

Compare its performance against the Phase 2 random agent using metrics such as:

* average score,
* median score,
* best score,
* average survival length.

---

## Phase 3 Learning Goals

By the end of this phase, understand:

* what a state is,
* what an action is,
* what a reward is,
* what an episode is,
* how a Q-table represents learned knowledge,
* why exploration is necessary,
* what epsilon means,
* what the learning rate controls,
* what the discount factor controls,
* and how the Q-learning update propagates future reward backward through earlier decisions.

---

# Phase 4 — Training Analysis and Visualization

## Goal

Make the learning process measurable rather than judging it only by watching the snake play.

## Add Tracking For

* score per episode,
* rolling average score,
* episode length,
* best score,
* epsilon,
* number of known states,
* and potentially reward per episode.

Plot training performance over time.

For example:

```text
Average score
^
|                 ________
|             ___/
|         ___/
|     ___/
|____/
+--------------------------> Episodes
```

## Learning Purpose

Understand:

* training curves,
* noisy performance,
* convergence,
* variance,
* whether learning has stalled,
* and whether a parameter change actually improved the system.

This phase should also make it easier to compare experiments.

---

# Phase 5 — Deep Q-Learning

## Goal

Replace the explicit Q-table with a neural network.

Instead of storing:

```text
state → Q-values
```

for every observed state, train a network to approximate:

```text
Q(state, action)
```

## Initial Network

A small network is sufficient.

For example:

```text
11 inputs
↓
Hidden layer
↓
Hidden layer
↓
3 outputs
```

The three outputs represent:

```text
Q(LEFT)
Q(STRAIGHT)
Q(RIGHT)
```

PyTorch can be introduced at this point.

## Concepts To Learn

This phase introduces:

* tensors,
* neural-network layers,
* forward passes,
* loss functions,
* gradient descent,
* backpropagation,
* optimizers,
* minibatches,
* experience replay,
* target values,
* and eventually target networks.

## Learning Purpose

The key conceptual change is:

```text
Q-table:
explicitly remembers individual states

Neural network:
learns a function that generalizes between states
```

---

# Phase 6 — Richer Observations

## Goal

Reduce the amount of hand-designed information given to the agent.

The original state representation tells the agent things such as:

```text
danger_left
food_up
```

These features were manually chosen because they seem useful.

A later version should receive more raw information about the game.

Possible approaches include:

* the complete board,
* a flattened grid,
* a local area around the snake,
* or separate grid channels for snake, food, and empty space.

## Learning Purpose

Explore the difference between:

```text
handcrafted features
```

and:

```text
features learned automatically from raw observations
```

The agent may now have enough information to recognize situations that the original 11-value state could not distinguish.

For example, it may eventually become capable of recognizing future traps caused by its own body rather than only immediate collisions.

---

# Phase 7 — Experimentation

## Goal

Use the completed system as a sandbox for exploring machine-learning behavior.

Change one variable at a time and observe how training changes.

## Possible Experiments

### Reward Shaping

Compare:

```text
Food: +10
Death: -10
```

against systems that also reward:

```text
moving toward food
```

or penalize:

```text
moving away from food
taking unnecessary steps
```

Observe whether faster learning necessarily produces better final behavior.

---

### Exploration

Experiment with:

```text
epsilon decay
minimum epsilon
no exploration
permanent exploration
```

Observe how insufficient exploration prevents useful strategies from being discovered.

---

### Discount Factor

Compare values such as:

```text
gamma = 0
gamma = 0.5
gamma = 0.9
gamma = 0.99
```

Observe how much the agent values delayed rewards.

---

### Learning Rate

Experiment with aggressive versus conservative learning rates.

Observe:

* slow learning,
* instability,
* and responsiveness to new experiences.

---

### State Representation

Try removing or adding information.

For example:

```text
remove current movement direction
```

or:

```text
add distance to food
```

Compare the resulting behavior.

---

### Network Design

Once using DQN, experiment with:

* hidden-layer size,
* number of layers,
* batch size,
* replay-buffer size,
* learning rate,
* and target-network update frequency.

---

# Guiding Principles

## Understand Before Abstracting

Avoid introducing large reinforcement-learning frameworks until the basic algorithms have been implemented manually.

The project is primarily for learning rather than producing the strongest possible Snake AI.

---

## Change One Thing at a Time

Machine-learning behavior can be difficult to diagnose when several parameters change simultaneously.

Prefer controlled experiments.

For example:

```text
Baseline:
gamma = 0.9

Experiment:
gamma = 0.5
```

while keeping everything else constant.

---

## Measure Results

Do not rely entirely on visually watching games.

Use statistics and evaluation runs to determine whether an experiment actually improved performance.

---

## Keep Training and Evaluation Separate

Training includes exploration.

Evaluation should normally disable exploration so the learned policy itself is being measured.

---

## Treat Strange Behavior as Useful Information

If an agent learns an unexpected strategy, first ask:

```text
What objective did I actually give it?
```

rather than:

```text
Why isn't it doing what I intended?
```

An agent optimizes the rewards and observations it receives, not the behavior the developer imagined.

Unexpected strategies are therefore useful demonstrations of how machine-learning objectives affect behavior.

---

# Overall Learning Path

The intended progression is:

```text
Human Snake
    ↓
Random Agent
    ↓
Environment / Episodes
    ↓
State Representation
    ↓
Rewards
    ↓
Tabular Q-Learning
    ↓
Training Analysis
    ↓
Neural Network / DQN
    ↓
Rawer Observations
    ↓
Experimentation
```

The project is successful if, by the end, it is possible to explain not only that the Snake agent learned, but also broadly **why and how the learning process works**.
