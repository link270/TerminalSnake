"""Small, optional JSONL experience recorder and organized run outputs."""

from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import platform
from uuid import uuid4

from checkpoints import agent_config, agent_kind
from snake import Action


STATE_FIELDS = ["danger_straight", "danger_right", "danger_left", "direction",
                "food_up", "food_down", "food_left", "food_right"]


def snapshot(game):
    return {"snake": list(game.snake), "food": game.food, "direction": game.direction,
            "score": game.score, "step": game.steps}


class EpisodeTrace:
    def __init__(self, stream, run_id, phase, episode, detailed):
        self.stream = stream
        self.identity = {"schema_version": 1, "run_id": run_id, "phase": phase, "episode": episode}
        self.detailed = detailed

    def write(self, event, **values):
        self.stream.write(json.dumps({**self.identity, "event": event, **values}) + "\n")


class RunOutput:
    def __init__(self, directory, agent, settings, record="none", record_episodes=5, record_every=1):
        if record not in ("none", "experience", "detailed") or record_episodes < 0 or record_every < 1:
            raise ValueError("Invalid recording settings")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.run_id = uuid4().hex
        self.record = record
        self.record_episodes = record_episodes
        self.record_every = record_every
        source = Path(__file__).parent
        metadata = {
            "schema_version": 1, "run_id": self.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "agent": agent_kind(agent), "agent_config": agent_config(agent),
            "settings": settings,
            "recording": {"level": record, "episodes_per_phase": record_episodes,
                          "every": record_every, "dqn_internal_records": False},
            "state_fields": STATE_FIELDS,
            "directions": {"0": "right", "1": "down", "2": "left", "3": "up"},
            "actions": [{"name": action.name.lower(), "value": action.value} for action in Action],
            "coordinates": "x increases right, y increases down; snake positions are head first",
            "rewards": {"food": 1, "collision": -1, "move": 0},
            "starvation_limit": "2 * grid_size ** 2 steps without food",
            "bootstrap_on_truncation": False,
            "episode_seeding": "one continuous environment RNG stream per phase, not per-episode reseeding",
            "python_version": platform.python_version(),
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in source.glob("*.py") if not p.name.startswith("test_")},
        }
        if agent_kind(agent) == "dqn":
            import torch

            metadata["torch_version"] = str(torch.__version__)
            metadata["network"] = {"layers": [11, 64, 64, 3], "hidden_activation": "relu"}
        (self.directory / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    @contextmanager
    def episode(self, phase, episode, local_episode):
        selected = (self.record != "none" and local_episode % self.record_every == 0
                    and (self.record_episodes == 0
                         or local_episode // self.record_every < self.record_episodes))
        if not selected:
            yield None
            return
        directory = self.directory / "experience" / phase
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / f"episode_{episode:08d}.jsonl").open("w", encoding="utf-8") as stream:
            yield EpisodeTrace(stream, self.run_id, phase, episode, self.record == "detailed")

    def write_results(self, phase, results):
        if not results:
            return
        with (self.directory / f"{phase}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
