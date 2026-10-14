from contextlib import nullcontext
from statistics import median
import sys

from agent_defs import RandomAgent, QLearningAgent, DQNAgent
from snake import SnakeGame, Action
from checkpoints import save_checkpoint, tuple_tree
from recording import snapshot


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


def run_episode(ep, runs, game, agent, verbose, train, trace=None):
    observation = game.reset()
    render_delay = 0.02
    steps_without_food = 0
    no_food_cap = 2 * game.size * game.size
    truncated = False
    done = game.done
    total_reward = 0
    reason = None
    epsilon_used = None if isinstance(agent, RandomAgent) else agent.epsilon
    if trace:
        trace.write("episode_start", board=snapshot(game), state=game.get_state(), epsilon=epsilon_used)
    while not done:
        state = game.get_state()
        before = snapshot(game) if trace else None
        decision = {} if trace and trace.detailed and not isinstance(agent, DQNAgent) else None
        update = {} if decision is not None and train and isinstance(agent, QLearningAgent) else None
        action = agent.choose_action(state, decision=decision)
        attempted_head = game.calculate_next_head(game.calculate_next_direction(action)) if trace else None
        observation, reward, done = game.step(action)
        next_state = None if done else game.get_state()
        total_reward += reward
        reason = None
        if done:
            if reward == -1:
                head = game.calculate_next_head(game.direction)
                reason = "wall_collision" if game.is_wall(head) else "body_collision"
            else:
                reason = "board_complete"
        
        if verbose >= 2:
            log = f"Episode: {ep}/{runs}, Step: {game.steps}, Snake pos:{observation[0]}, Food pos: {observation[1]} Reward: {reward}, Done: {done}"
            if not game.render(render_delay, log):
                print(log)
        else:
            game.render(render_delay)

        steps_without_food = 0 if reward == 1 else steps_without_food + 1
        if not done and steps_without_food >= no_food_cap:
            print(f"\nRun: {ep} went {steps_without_food} without food. Ending early.")
            truncated = True
            done = True
            reason = "starvation_limit"

        if train:
            agent.learn(state, action, reward, next_state, done, update=update)
        if trace:
            trace.write(
                "transition", step=game.steps, before=before, after=snapshot(game),
                state=state, next_state=next_state, action=action.value,
                attempted_head=attempted_head, reward=reward, terminated=game.done,
                truncated=truncated, done_for_learning=done, termination_reason=reason,
                decision=decision, update=update,
            )

    if train: agent.decay()
    outcome = {"truncated": truncated, "terminated": game.done, "termination_reason": reason,
               "total_reward": total_reward, "epsilon_used": epsilon_used}
    if trace:
        trace.write("episode_end", **outcome, steps=game.steps, score=game.score,
                    epsilon_after=None if isinstance(agent, RandomAgent) else agent.epsilon)
    return outcome


def run_episodes(size=10, runs=1000, render=False, verbose=0, agent=None, game_seed=None,
                 train=False, progress_bar=False, output=None, checkpoint_every=None,
                 resume=None):
    if agent is None:
        print("Agent cannot be none.")
        return []

    if runs <=0:
        print("Skipping due to insufficient runs.")
        return []

    game = SnakeGame(size=size, render=render, seed=game_seed)
    start_episode = 0
    if resume:
        if not train or size != resume["environment"]["size"] or game_seed != resume["environment"]["seed"]:
            raise ValueError("Resuming requires training with the saved environment size and seed")
        game.random.setstate(tuple_tree(resume["environment"]["rng"]))
        start_episode = resume["episodes_completed"]
    phase = "training" if train else "evaluation"
    if checkpoint_every is not None and output is None:
        raise ValueError("Checkpoint saving requires a run output directory")
    if train and checkpoint_every is not None:
        save_checkpoint(output.directory / "checkpoints", agent, game, start_episode)

    if progress_bar: progress_interval = max(1, runs // 100) 
    results = []
    for local_ep in range(runs):
        ep = start_episode + local_ep
        context = output.episode(phase, ep, local_ep) if output else nullcontext(None)
        with context as trace:
            outcome = run_episode(ep, start_episode + runs, game, agent, verbose, train, trace)

        if progress_bar:
            if local_ep % progress_interval == 0 or local_ep == runs - 1:
                percent = int((local_ep + 1) / runs * 100)
                bar_length = 40
                filled_length = int(bar_length * percent // 100)
                bar = '#' * filled_length + '-' * (bar_length - filled_length)
                
                # \r moves the cursor back to the start of the line
                sys.stdout.write(f'\rProgress: |{bar}| {percent}% | Run: {ep+1}/{start_episode + runs}')
                sys.stdout.flush()

        
        result = {
            "episode": ep,
            "grid_size": size,
            "max_possible_score": size ** 2 - 2,
            "score": game.score,
            "steps": game.steps,
            **outcome,
            }
        
        if not isinstance(agent, RandomAgent):
            result["epsilon"] = agent.epsilon
            if isinstance(agent, QLearningAgent):
                result["num_q_table_states"] = len(agent.q_table)
        
        results.append(result)
        if train and checkpoint_every is not None:
            if local_ep == runs - 1 or (checkpoint_every > 0 and (ep + 1) % checkpoint_every == 0):
                save_checkpoint(output.directory / "checkpoints", agent, game, ep + 1)
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
    from agent_cli import main

    main()
