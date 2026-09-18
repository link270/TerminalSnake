import argparse
import random
import sys
from abc import ABC, abstractmethod

from snake import SnakeGame, Action

MAX_STEPS = 1000

class Agent(ABC):
    def __init__(self, seed=None):
        if seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = seed

        self.rng = random.Random(self.seed)

    @abstractmethod
    def choose_action(self, state) -> Action:
        pass

    @abstractmethod
    def learn(self, state, action, reward, next_state, done,):
        pass

    def decay(self):
        pass


class RandomAgent(Agent):
    def choose_action(self, state) -> Action:
        return self.rng.choice(list(Action))
    
    def learn(self, state, action, reward, next_state, done,):
        pass


class QLearningAgent(Agent):
    def __init__(self, seed=None, epsilon=1.0, epsilon_decay = 0.995, minimum_epsilon = 0.05, alpha = 0.1, gamma = 0.9):
        super().__init__(seed)
        self.q_table = {}

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))

        q_values = self.get_q_values(state)
        best_value = max(q_values)
        best_indices = [index for index, value in enumerate(q_values) if value == best_value]
        best_index = self.rng.choice(best_indices)
        return list(Action)[best_index]

    def get_q_values(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0]

        return self.q_table[state]

    # new Q = old Q + α × (reward + γ × best future Q - old Q)
    def learn(self, state, action, reward, next_state, done):
        action_index = list(Action).index(action)
        q_values = self.get_q_values(state)
        old_value = q_values[action_index]
        best_future_q = 0 if done else max(self.get_q_values(next_state))

        new_q = old_value + self.alpha * (reward + self.gamma * best_future_q - old_value)
        q_values[action_index] = new_q

    def decay(self):
        self.epsilon = max(self.minimum_epsilon, self.epsilon * self.epsilon_decay)



def compile_results(results):
    if not results:
        raise ValueError("results cannot be empty")
    best_score = max(results, key=lambda result: result["score"])
    best_survival = max(results, key=lambda result: result["steps"])
    return {
        "average_score": sum(result["score"] for result in results) / len(results),
        "best_score": best_score["score"],
        "best_score_episode": best_score["episode"],
        "average_survival": sum(result["steps"] for result in results) / len(results),
        "best_survival": best_survival["steps"],
        "best_survival_episode": best_survival["episode"],
    }


def run_episode(ep, runs, game, agent, verbose):
    observation = game.reset()
    render_delay = 0.02
    while not game.done:
        state = game.get_state()
        action = agent.choose_action(state)
        observation, reward, done = game.step(action)

        next_state = None if done else game.get_state()

        agent.learn(state, action, reward, next_state, done)
        
        if verbose >= 2:
            log = f"Episode: {ep}/{runs}, Step: {game.steps}, Snake pos:{observation[0]}, Food pos: {observation[1]} Reward: {reward}, Done: {done}"
            if not game.render(render_delay, log):
                print(log)
        else:
            game.render(render_delay)

        if game.steps >= MAX_STEPS:
            print(f"Run went over the max steps of {MAX_STEPS}")
            break

    agent.decay()


def begin(size=10, runs=1000, render=False, verbose=0, agent = None, game_seed=None):
    if agent is None:
        print("Agent cannot be none.")
        return

    game = SnakeGame(size=size, render=render, seed=game_seed)

    results = []
    for ep in range(runs):
        run_episode(ep, runs, game, agent, verbose)
        result = {"episode": ep, "score": game.score, "steps": game.steps}
        results.append(result)
        if not render and verbose >= 1:
            print(result)

    if render and verbose >= 1:
        print(*results, sep="\n")
    summary = compile_results(results)
    print(
        f"Average score: {summary['average_score']:.2f}\n"
        f"Best score: {summary['best_score']} (episode {summary['best_score_episode']})\n"
        f"Average survival: {summary['average_survival']:.2f} steps\n"
        f"Best survival: {summary['best_survival']} steps (episode {summary['best_survival_episode']})\n"
        f"Agent seed: {agent.seed} Game seed: {game.seed}"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a game of snake with an agent.")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--agent_seed", type=int, default=None)
    parser.add_argument("--game_seed", type=int, default=None)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--agent", choices=("random", "q_learning"), default=None)
    args = parser.parse_args()

    agent = QLearningAgent(seed=args.agent_seed) if args.agent == "q_learning" else RandomAgent(seed=args.agent_seed)
    begin(args.size, args.runs, args.render, args.verbose, agent, args.game_seed)
