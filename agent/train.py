from collections import deque

import numpy as np
import torch
import requests
from dqn_agent import Agent
from model import QNetwork
from rich import print
from rich.traceback import install as install_traceback

install_traceback()


class Env:
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    def __init__(self, protocol="http", host="localhost", port=8881):
        self.url = f"{protocol}://{host}:{port}"
        self.json = None
        self.board = None

    def read(self):
        self.json = requests.get(f"{self.url}/state").json()
        self._read_board()

    def reset(self):
        self.json = requests.put(f"{self.url}/reset").json()
        self._read_board()

    def step(self, action: int):
        self.json = requests.put(f"{self.url}/action", data=Env.ACTIONS[action]).json()
        self._read_board()

    @property
    def state(self):
        """Converts board values (0, 2, 4, ..., 2 ** 16) to indexes (0, 1, ..., 16)"""
        b = self.board.copy()
        b[self.board == 0] = 1
        return np.log2(b).astype(int)

    @property
    def reward(self):
        """Penalize forbidden moves with -1"""
        if self.done:
            return -10
        return -1 if self.invalid else self.delta_score

    @property
    def done(self):
        return self.json['done']

    @property
    def score(self):
        return self.json['score']

    @property
    def delta_score(self):
        return self.json['deltaScore']

    @property
    def invalid(self):
        return self.json['invalid']

    @property
    def human(self):
        return self.json['humanBoard']

    @property
    def max_tile(self):
        return self.board.max()

    def _read_board(self):
        self.board = np.array([[e['value'] if e is not None else 0 for e in r] for r in self.json['board']])


def dqn(env, agent, n_episodes=1000, max_t=10000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    rolling_scores = 0
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env.reset()  # reset the environment
        state = env.state  # get the current state
        score = 0
        t = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env.step(action)
            next_state = env.state
            reward = env.reward
            done = env.done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        rolling_scores = rolling_scores * 0.95 + score * 0.05 if i_episode > 1 else score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            '[yellow on default]Episode {}[/]  \tStp: {}  \tScore: {} / {:.0f} \tAvg: {:.2f}  \tTile: {}  \tEps: {:.3f}  \tmem: {}  '
                .format(i_episode, t + 1, env.score, score, rolling_scores, env.max_tile, eps,
                        len(agent.memory)), end="\r")
        if i_episode % 10 == 0:
            print("")

        # Save trained parameters
        if i_episode % 100 == 0:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    return rolling_scores


def train():
    env = Env()
    env.reset()
    state = env.state
    agent = Agent(QNetwork, len(state), len(Env.ACTIONS), seed=42)
    print(agent.__dict__)

    # Load trained parameters
    # agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    # agent.qnetwork_target.load_state_dict(torch.load('checkpoint.pth'))

    scores = dqn(env, agent, n_episodes=20_000, eps_decay=0.995, eps_start=1., eps_end=0.01)

    # Save trained parameters
    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')


if __name__ == "__main__":
    train()
