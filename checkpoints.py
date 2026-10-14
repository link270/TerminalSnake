"""Agent snapshots, including enough state to resume at the next episode."""

import json
from pathlib import Path

from agent_defs import DQNAgent, QLearningAgent, RandomAgent


AGENTS = {"random": RandomAgent, "q_learning": QLearningAgent, "dqn": DQNAgent}
VERSION = 1


def agent_kind(agent):
    return next(name for name, cls in AGENTS.items() if isinstance(agent, cls))


def agent_config(agent):
    names = ["seed"]
    if not isinstance(agent, RandomAgent):
        names += ["epsilon", "epsilon_decay", "minimum_epsilon", "gamma"]
        names += ["alpha"] if isinstance(agent, QLearningAgent) else ["learning_rate"]
    return {name: getattr(agent, name) for name in names}


def tuple_tree(value):
    """JSON arrays back to the tuples required by random.setstate()."""
    return tuple(tuple_tree(item) for item in value) if isinstance(value, list) else value


def save_checkpoint(directory, agent, game, episodes_completed):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": VERSION,
        "agent": agent_kind(agent), "config": agent_config(agent),
        "agent_rng": agent.rng.getstate(),
        "episodes_completed": episodes_completed,
        "environment": {"size": game.size, "seed": game.seed, "rng": game.random.getstate(),
                        "diagonal_obstacles": game.diagonal_obstacles},
        "boundary": "before_first_episode" if episodes_completed == 0 else "after_episode_and_epsilon_decay",
    }
    if isinstance(agent, DQNAgent):
        import torch

        data.update(model=agent.model.state_dict(), optimizer=agent.optimizer.state_dict(),
                    torch_rng=torch.get_rng_state())
        path = directory / f"episode_{episodes_completed:08d}.pt"
        torch.save(data, path)
    else:
        if isinstance(agent, QLearningAgent):
            data["q_table"] = [{"state": state, "values": values} for state, values in agent.q_table.items()]
        path = directory / f"episode_{episodes_completed:08d}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_checkpoint(path):
    path = Path(path)
    if path.suffix == ".pt":
        import torch

        data = torch.load(path, map_location="cpu", weights_only=True)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
    if data["schema_version"] != VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    diagonal = data["environment"].get("diagonal_obstacles", False)
    if not isinstance(diagonal, bool) or (diagonal and data["agent"] == "dqn"):
        raise ValueError("Unsupported checkpoint observation mode")
    if data["agent"] == "q_learning":
        width = 12 if diagonal else 8
        if any(len(row["state"]) != width for row in data["q_table"]):
            raise ValueError("Q-table states do not match the checkpoint observation mode")
    agent = AGENTS[data["agent"]](**data["config"])
    agent.diagonal_obstacles = diagonal
    agent.rng.setstate(tuple_tree(data["agent_rng"]))
    if isinstance(agent, QLearningAgent):
        agent.q_table = {tuple(row["state"]): row["values"] for row in data["q_table"]}
    elif isinstance(agent, DQNAgent):
        import torch

        agent.model.load_state_dict(data["model"])
        agent.optimizer.load_state_dict(data["optimizer"])
        torch.set_rng_state(data["torch_rng"])
    return agent, data
