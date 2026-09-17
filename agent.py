import argparse
import random
import sys
from abc import ABC, abstractmethod

from snake import SnakeGame, Action


class Agent(ABC):
    def __init__(self, seed=None):
        if seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = seed

        self.rng = random.Random(self.seed)

    @abstractmethod
    def choose_action(self, observation) -> Action:
        pass


class RandomAgent(Agent):
    def choose_action(self, observation) -> Action:
        return self.rng.choice(list(Action))


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
        action = agent.choose_action(observation)
        observation, reward, done = game.step(action)
        if verbose >= 2:
            log = f"Episode: {ep}/{runs}, Step: {game.steps}, Snake pos:{observation[0]}, Food pos: {observation[1]} Reward: {reward}, Done: {done}"
            if not game.render(render_delay, log):
                print(log)
        else:
            game.render(render_delay)


def begin(size=10, runs=1000, render=False, verbose=0, agent: Agent = None, game_seed=None):
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
    parser.add_argument("--agent", choices=("random",), default=None)
    args = parser.parse_args()

    agent = RandomAgent(seed=args.agent_seed) if args.agent else RandomAgent(seed=args.agent_seed)
    begin(args.size, args.runs, args.render, args.verbose, agent, args.game_seed)
