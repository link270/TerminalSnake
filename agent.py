import argparse
import random
from abc import ABC, abstractmethod

from snake import SnakeGame, Action


class Agent(ABC):
    @abstractmethod
    def choose_action(self) -> Action:
        pass


class RandomAgent(Agent):
    def choose_action(self) -> Action:
        return random.choice(list(Action))

def run_episode(ep, runs, game, agent, verbose):
    observation = game.reset()
    render_delay = 0.02
    while not game.done:
        action = agent.choose_action()
        observation, reward, done = game.step(action)
        if verbose:
            log = f"Episode: {ep}/{runs}, Step: {game.steps}, Snake pos:{observation[0]}, Food pos: {observation[1]} Reward: {reward}, Done: {done}"
            if not game.render(render_delay, log):
                print(log)
        else:
            game.render(render_delay)


def begin(size=10, runs=1000, render=False, verbose=False, agent: Agent | None = None):
    if agent is None:
        agent = RandomAgent()
    game = SnakeGame(size=size, render=render)

    results = []
    for ep in range(runs):
        run_episode(ep, runs, game, agent, verbose)
        result = {"episode": ep, "score": game.score, "steps": game.steps}
        results.append(result)
        if not render:
            print(result)

    if render:
        print(*results, sep="\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a game of snake with an agent.")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--agent", choices=("random",), default=None)
    args = parser.parse_args()
    begin(args.size, args.runs, args.render, args.verbose, RandomAgent() if args.agent else None)
