import random
import sys

from abc import ABC, abstractmethod
from snake import  Action

class Agent(ABC):
    def __init__(self, seed=None):
        if seed is None:
            self.seed = random.randrange(sys.maxsize)
        else:
            self.seed = seed

        self.rng = random.Random(self.seed)

        self.q_table = {}
        self.epsilon = 'N/A'
        self.epsilon_decay = 'N/A'
        self.minimum_epsilon = 'N/A'
        self.alpha = 'N/A'
        self.gamma = 'N/A'

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

class DNQAgent(Agent):
    def __init__(self, seed=None, epsilon=1.0, epsilon_decay = 0.995, minimum_epsilon = 0.05, gamma = 0.9, learning_rate=0.001,):
        super().__init__(seed)

        self.model
        self.optimizer

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.minimum_epsilon = minimum_epsilon
        self.gamma = gamma
        self.learning_rate = learning_rate

    def choose_action(self, state) -> Action:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(Action))

        q_values = self.get_q_values(state)
        best_value = max(q_values)
        best_indices = [index for index, value in enumerate(q_values) if value == best_value]
        best_index = self.rng.choice(best_indices)
        return list(Action)[best_index]

    def encode_state(self, state):
        encoded_dir = [int(state[3] == direction) for direction in range(4)]
        
        return(
            int(state[0]),
            int(state[1]),
            int(state[2]),
            encoded_dir[0],
            encoded_dir[1],
            encoded_dir[2],
            encoded_dir[3],
            int(state[4]),
            int(state[5]),
            int(state[6]),
            int(state[7]),
        )

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