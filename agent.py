import argparse
import random
from statistics import median
import sys
import csv

from pathlib import Path
from agent_defs import RandomAgent, QLearningAgent, DQNAgent
from snake import SnakeGame, Action


def compile_results(results):
    if not results:
        raise ValueError("results cannot be empty")
    best_score = max(results, key=lambda result: result["score"])
    best_survival = max(results, key=lambda result: result["steps"])
    return {
        "average_score": sum(result["score"] for result in results) / len(results),
        "median_score": median(result["score"] for result in results),
        "best_score": best_score["score"],
        "best_score_episode": best_score["episode"],
        "average_survival": sum(result["steps"] for result in results) / len(results),
        "best_survival": best_survival["steps"],
        "best_survival_episode": best_survival["episode"],
        "max_possible_score": results[0]["max_possible_score"],
    }


def run_episode(ep, runs, game, agent, verbose, train):
    observation = game.reset()
    render_delay = 0.02
    steps_without_food = 0
    no_food_cap = 2 * game.size * game.size
    truncated = False
    done = game.done
    while not done:
        state = game.get_state()
        action = agent.choose_action(state)
        observation, reward, done = game.step(action)
        next_state = None if done else game.get_state()
        
        if verbose >= 2:
            log = f"Episode: {ep}/{runs}, Step: {game.steps}, Snake pos:{observation[0]}, Food pos: {observation[1]} Reward: {reward}, Done: {done}"
            if not game.render(render_delay, log):
                print(log)
        else:
            game.render(render_delay)

        steps_without_food = 0 if reward == 1 else steps_without_food + 1
        if steps_without_food >= no_food_cap:
            print(f"\nRun: {ep} went {steps_without_food} without food. Ending early.")
            truncated = True
            done = True

        if train: agent.learn(state, action, reward, next_state, done)

    if train: agent.decay()
    return truncated


def run_episodes(size=10, runs=1000, render=False, verbose=0, agent = None, game_seed=None, train=False, progress_bar=False):
    if agent is None:
        print("Agent cannot be none.")
        return

    if runs <=0:
        print("Skipping due to insufficient runs.")
        return

    game = SnakeGame(size=size, render=render, seed=game_seed)

    if progress_bar: progress_interval = max(1, runs // 100) 
    results = []
    for ep in range(runs):
        truncated = run_episode(ep, runs, game, agent, verbose, train)

        if progress_bar:
            if ep % progress_interval == 0 or ep == runs - 1:
                percent = int((ep + 1) / runs * 100)
                bar_length = 40
                filled_length = int(bar_length * percent // 100)
                bar = '█' * filled_length + '-' * (bar_length - filled_length)
                
                # \r moves the cursor back to the start of the line
                sys.stdout.write(f'\rProgress: |{bar}| {percent}% | Run: {ep+1}/{runs}')
                sys.stdout.flush()

        
        result = {
            "episode": ep,
            "grid_size": size,
            "max_possible_score": size ** 2 - 2,
            "score": game.score,
            "steps": game.steps,
            "truncated": truncated,
            }
        
        if not isinstance(agent, RandomAgent):
            result["epsilon"] = agent.epsilon
            if isinstance(agent, QLearningAgent):
                result["num_q_table_states"] = len(agent.q_table)
        
        results.append(result)
        if not render and verbose >= 1:
            print(result)

    if train:
        for i in range(len(results)):
            length = min(100, i+1)
            rolling_count = 0
            for r in range(max(0, i-99), i+1):
                rolling_count += results[r]["score"]

            results[i]["rolling_average"] = rolling_count / length

    if progress_bar:
        sys.stdout.write('\n\n')
        sys.stdout.flush()
    
    if render and verbose >= 1:
        print(*results, sep="\n")
    summary = compile_results(results)
    print(
        f"Average score: {summary['average_score']:.2f}\n"
        f"Median score: {summary['median_score']}\n"
        f"Best score: {summary['best_score']} (episode {summary['best_score_episode']})\n"
        f"Best possible score: {summary['max_possible_score']}\n"
        f"Average survival: {summary['average_survival']:.2f} steps\n"
        f"Best survival: {summary['best_survival']} steps (episode {summary['best_survival_episode']})\n"
        f"Agent seed: {agent.seed} Game seed: {game.seed}"
    )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start a game of snake with an agent.")
    parser.add_argument("--size", type=int, default=10)
    parser.add_argument("--agent_seed", type=int, default=None)
    parser.add_argument("--game_seed", type=int, default=None)
    parser.add_argument("--render_eval", action="store_true")
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--agent", choices=("random", "q_learning", "dqn"), default="q_learning")
    parser.add_argument("-tr", "--training_runs", type=int, default=10_000)
    parser.add_argument("-er", "--evaluation_runs", type=int, default=1_000)
    args = parser.parse_args()

    match args.agent:
        case "random":
            agent = RandomAgent(seed=args.agent_seed)
        case "q_learning":
            agent = QLearningAgent(seed=args.agent_seed)
        case "dqn":
            agent = DQNAgent(seed=args.agent_seed)
    
    training_runs = args.training_runs
    eval_runs = args.evaluation_runs

    print(
        f"Starting with agent: {args.agent}, grid size: {args.size}, verbose level: {args.verbose}\n"
        f"rendering: Training: {args.render_train} | Eval: {args.render_eval}\n"
        f"{f"Training with {training_runs} runs.\n" if training_runs > 0 else ""}"
        f"{f"Evaluating with {eval_runs} runs.\n" if eval_runs > 0 and args.agent !="random" else ""}"
        f"\nGame seed: {"Random" if args.game_seed is None else args.game_seed}\n"
        f"Agent seed: {"Random" if args.agent_seed is None else args.agent_seed}"
        )

    if args.game_seed is None:
        game_seed = random.randrange(sys.maxsize)
    else:
        game_seed = args.game_seed

    if args.agent != "random":
        print(f"\n\nStarting {training_runs} training runs.\n")

        training_results = run_episodes(args.size, training_runs, args.render_train, args.verbose, agent, game_seed, train=True, progress_bar=True)
        
        if not Path("Output").is_dir():
            Path.mkdir("Output")
    
        with open(f"Output/training_results_{training_runs}_{game_seed}_{args.agent}.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=training_results[0].keys())
            writer.writeheader()
            writer.writerows (training_results)

        agent.epsilon = 0.0
        
    print(f"\n\nStarting {eval_runs} evaluation runs.\n")

    eval_results = run_episodes(args.size, eval_runs, args.render_eval, args.verbose, agent, game_seed)
    with open(f"Output/evaluation_results_{training_runs}_{game_seed}_{args.agent}.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=eval_results[0].keys())
        writer.writeheader()
        writer.writerows (eval_results)