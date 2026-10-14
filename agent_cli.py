"""Command-line configuration and run lifecycle, separate from the game loop."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import sys
from uuid import uuid4

from checkpoints import AGENTS, agent_kind, load_checkpoint
from recording import RunOutput


def main(argv=None):
    from agent import run_episodes

    parser = argparse.ArgumentParser(description="Train, evaluate and optionally record Snake agents.")
    parser.add_argument("--agent", choices=AGENTS, default=None)
    parser.add_argument("--diagonal-obstacles", action=argparse.BooleanOptionalAction, default=None,
                        help="Append four diagonal obstacle flags (Q-learning/random only); off for new agents")
    parser.add_argument("--size", type=int, default=None)
    parser.add_argument("--agent_seed", type=int)
    parser.add_argument("--game_seed", type=int)
    parser.add_argument("--eval-seed", type=int, help="Evaluation environment seed; defaults to game_seed + 1")
    parser.add_argument("--eval-agent-seed", type=int, default=0, help="Evaluation action/tie-breaking seed")
    parser.add_argument("--render_eval", action="store_true")
    parser.add_argument("--render_train", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("-tr", "--training_runs", type=int, default=None,
                        help="Additional training episodes; defaults to 10000, or 0 for load-only/random")
    parser.add_argument("-er", "--evaluation_runs", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, help="New run directory; must not already exist")
    parser.add_argument("--record", choices=("none", "experience", "detailed"), default="none")
    parser.add_argument("--record-episodes", type=int, default=5,
                        help="Maximum recorded episodes per phase; 0 means all (default: 5)")
    parser.add_argument("--record-every", type=int, default=1, help="Record every Nth episode, starting with the first")
    parser.add_argument("--save-checkpoints", action="store_true", help="Save resumable initial and final training checkpoints")
    parser.add_argument("--checkpoint-every", type=int, default=0,
                        help="Also save every N completed training episodes; implies --save-checkpoints")
    parser.add_argument("--load-checkpoint", type=Path)
    parser.add_argument("--resume-training", action="store_true", help="Resume training from --load-checkpoint")
    args = parser.parse_args(argv)
    if args.record_episodes < 0 or args.record_every < 1 or args.checkpoint_every < 0:
        parser.error("Recording limits must be nonnegative and --record-every must be positive")
    if args.resume_training and not args.load_checkpoint:
        parser.error("--resume-training requires --load-checkpoint")

    checkpoint = None
    if args.agent == "dqn" and args.diagonal_obstacles:
        parser.error("--diagonal-obstacles supports Q-learning and random agents only")
    if args.load_checkpoint:
        agent, checkpoint = load_checkpoint(args.load_checkpoint)
        if args.diagonal_obstacles is not None and args.diagonal_obstacles != agent.diagonal_obstacles:
            parser.error("Observation mode must match the checkpoint; omit the flag to use its saved mode")
        if args.agent is not None and args.agent != agent_kind(agent):
            parser.error("--agent does not match the checkpoint")
        if args.agent_seed is not None:
            parser.error("A loaded agent keeps its saved RNG; use --eval-agent-seed for evaluation")
        size = checkpoint["environment"]["size"] if args.size is None else args.size
        game_seed = checkpoint["environment"]["seed"] if args.game_seed is None else args.game_seed
        if args.resume_training and (size != checkpoint["environment"]["size"]
                                     or game_seed != checkpoint["environment"]["seed"]):
            parser.error("Resuming training requires the saved --size and --game_seed")
    else:
        agent = AGENTS[args.agent or "q_learning"](seed=args.agent_seed)
        agent.diagonal_obstacles = bool(args.diagonal_obstacles)
        size = 10 if args.size is None else args.size
        game_seed = random.randrange(sys.maxsize) if args.game_seed is None else args.game_seed
    training_runs = args.training_runs
    if training_runs is None:
        training_runs = 0 if (checkpoint and not args.resume_training) or agent_kind(agent) == "random" else 10000
    if checkpoint and training_runs > 0 and not args.resume_training:
        parser.error("Use --resume-training to continue learning from a checkpoint")
    if size < 3 or training_runs < 0 or args.evaluation_runs < 0:
        parser.error("Size must be at least 3 and episode counts must be nonnegative")
    if training_runs + args.evaluation_runs == 0:
        parser.error("Request at least one training or evaluation episode")
    save = args.save_checkpoints or args.checkpoint_every > 0
    if (save or args.resume_training) and training_runs == 0:
        parser.error("Saving/resuming training checkpoints requires training episodes")
    eval_seed = game_seed + 1 if args.eval_seed is None else args.eval_seed
    directory = args.output_dir or Path("Output") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                                                     + "_" + agent_kind(agent) + "_" + uuid4().hex[:8])
    if directory.exists():
        parser.error("--output-dir must be a new directory to avoid overwriting a run")
    settings = {
        "size": size, "game_seed": game_seed, "evaluation_seed": eval_seed,
        "evaluation_agent_seed": args.eval_agent_seed,
        "training_runs": training_runs, "evaluation_runs": args.evaluation_runs,
        "source_checkpoint": str(args.load_checkpoint.resolve()) if checkpoint else None,
        "source_checkpoint_episodes": checkpoint["episodes_completed"] if checkpoint else None,
        "evaluation_checkpoint_episode": (checkpoint["episodes_completed"] if checkpoint else 0) + training_runs,
        "resume_training": args.resume_training,
        "checkpoint_every": args.checkpoint_every if save else None,
    }
    output = RunOutput(directory, agent, settings, args.record, args.record_episodes, args.record_every)
    print(f"Run output: {directory.resolve()}")
    if training_runs:
        results = run_episodes(size, training_runs, args.render_train, args.verbose, agent, game_seed,
                               train=True, progress_bar=True, output=output,
                               checkpoint_every=args.checkpoint_every if save else None,
                               resume=checkpoint if args.resume_training else None)
        output.write_results("training", results)
    if args.evaluation_runs:
        # Evaluation uses independent randomness and does not consume training state.
        epsilon, rng = agent.epsilon, agent.rng.getstate()
        agent.epsilon = 0.0
        agent.rng.seed(args.eval_agent_seed)
        try:
            results = run_episodes(size, args.evaluation_runs, args.render_eval, args.verbose,
                                   agent, eval_seed, output=output, progress_bar=True)
            output.write_results("evaluation", results)
        finally:
            agent.epsilon = epsilon
            agent.rng.setstate(rng)


if __name__ == "__main__":
    main()
