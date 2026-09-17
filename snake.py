import argparse
from enum import Enum
import os
import random
import sys
import time
from typing import NamedTuple


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))

class Action(Enum):
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1


class StepResult(NamedTuple):
    observation: tuple
    reward: int
    done: bool


class SnakeGame:
    """Small Snake environment. Actions are -1 (left), 0 (straight), 1 (right)."""

    def __init__(self, size=10, render=True, seed=None):
        if size < 3:
            raise ValueError("size must be at least 3")
        self.size = size
        self.render_enabled = render

        if seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = seed

        self.random = random.Random(self.seed)
        self.reset()

    def reset(self):
        middle = self.size // 2
        self.snake = [(middle, middle), (middle - 1, middle)]
        self.direction = 0
        self.score = 0
        self.steps = 0
        self.done = False
        self._place_food()
        return self.observation

    @property
    def observation(self):
        return tuple(self.snake), self.food

    def calculate_next_head(self, direction):
        dx, dy = DIRECTIONS[direction]
        head = self.snake[0]
        return (head[0] + dx, head[1] + dy)

    def calculate_next_direction(self, action: Action):
        return (self.direction + action.value) % 4

    def is_wall(self, position):
        px = position[0]
        py = position[1]
        return not (0 <= px < self.size and 0 <= py < self.size)

    def is_snake(self, position):
        return position in self.snake[:-1]

    def is_collision(self, position):
        return self.is_wall(position) or self.is_snake(position)

    def would_collide(self, action: Action):
        direction = self.calculate_next_direction(action)
        return self.is_collision(self.calculate_next_head(direction))

    def step(self, action: Action = Action.STRAIGHT):
        if self.done:
            raise RuntimeError("reset the game before stepping again")
        if action not in Action:
            raise ValueError("action must be -1 (left), 0 (straight), or 1 (right)")

        self.steps += 1
        self.direction = self.calculate_next_direction(action)
        new_head = self.calculate_next_head(self.direction)
        if self.is_collision(new_head):
            self.done = True
            return StepResult(self.observation, -1, self.done)

        ate = new_head == self.food

        self.snake.insert(0, new_head)
        if ate:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        return StepResult(self.observation, 1 if ate else 0, self.done)

    def _place_food(self):
        empty = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in self.snake]
        self.food = self.random.choice(empty) if empty else None
        if not empty:
            self.done = True

    def render(self, delay=0, message=None):
        if not self.render_enabled:
            return False

        time.sleep(delay)
        cells = set(self.snake[1:])
        rows = []
        for y in range(self.size):
            rows.append("".join("@" if (x, y) == self.snake[0] else "*" if (x, y) == self.food else "o" if (x, y) in cells else " " for x in range(self.size)))
        print("\x1b[H\x1b[J" + f"Score: {self.score}\n+{'-' * self.size}+\n" + "\n".join(f"|{row}|" for row in rows) + f"\n+{'-' * self.size}+" + (f"\n{message}" if message is not None else ""), flush=True)
        return True


def _read_key():
    if os.name == "nt":
        import msvcrt

        if not msvcrt.kbhit():
            return None
        key = msvcrt.getwch()
        return msvcrt.getwch() if key in ("\x00", "\xe0") else key


def play(size=10, delay=0.15):
    game = SnakeGame(size=size)

    keys = {"K": 2, "M": 0, "H": 3, "P": 1}

    while not game.done:
        game.render(delay)
        key = _read_key()
        if key in ("q", "Q"):
            return
        action = Action.STRAIGHT
        if key in keys:
            wanted = keys[key]
            turn = (wanted - game.direction) % 4
            if turn in (1, 3):
                action = Action.RIGHT if turn == 1 else Action.LEFT
        game.step(action)
    game.render()
    print("Game over!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Snake with the arrow keys; Q quits.")
    parser.add_argument("--size", type=int, default=10)
    args = parser.parse_args()
    play(args.size)
