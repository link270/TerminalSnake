import argparse
from enum import Enum
import os
import random
import sys
import time


DIRECTIONS = ((1, 0), (0, 1), (-1, 0), (0, -1))

class Action(Enum):
    LEFT = -1
    STRAIGHT = 0
    RIGHT = 1

class SnakeGame:
    """Small Snake environment. Actions are -1 (left), 0 (straight), 1 (right)."""

    def __init__(self, size=10, render=True, seed=None):
        if size < 3:
            raise ValueError("size must be at least 3")
        self.size = size
        self.render_enabled = render
        self.random = random.Random(seed)
        self.reset()

    def reset(self):
        middle = self.size // 2
        self.snake = [(middle, middle), (middle - 1, middle)]
        self.direction = 0
        self.score = 0
        self.done = False
        self._place_food()
        return self.observation

    @property
    def observation(self):
        return tuple(self.snake), self.food

    def step(self, action=Action.STRAIGHT):
        if self.done:
            raise RuntimeError("reset the game before stepping again")
        if action not in Action:
            raise ValueError("action must be -1 (left), 0 (straight), or 1 (right)")

        self.direction = (self.direction + action.value) % 4
        dx, dy = DIRECTIONS[self.direction]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)
        ate = new_head == self.food
        body = self.snake if ate else self.snake[:-1]

        if not (0 <= new_head[0] < self.size and 0 <= new_head[1] < self.size) or new_head in body:
            self.done = True
            return self.observation, -1, True

        self.snake.insert(0, new_head)
        if ate:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        return self.observation, 1 if ate else 0, self.done

    def _place_food(self):
        empty = [(x, y) for y in range(self.size) for x in range(self.size) if (x, y) not in self.snake]
        self.food = self.random.choice(empty) if empty else None
        if not empty:
            self.done = True

    def render(self):
        if not self.render_enabled:
            return
        cells = set(self.snake[1:])
        rows = []
        for y in range(self.size):
            rows.append("".join("@" if (x, y) == self.snake[0] else "*" if (x, y) == self.food else "o" if (x, y) in cells else " " for x in range(self.size)))
        print("\x1b[H\x1b[J" + f"Score: {self.score}\n+{'-' * self.size}+\n" + "\n".join(f"|{row}|" for row in rows) + f"\n+{'-' * self.size}+", flush=True)


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
        game.render()
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
        time.sleep(delay)
    game.render()
    print("Game over!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Snake with the arrow keys; Q quits.")
    parser.add_argument("--size", type=int, default=10)
    args = parser.parse_args()
    play(args.size)
