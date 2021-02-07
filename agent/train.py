from collections import deque

import numpy as np
import torch
import requests
from dqn_agent import Agent
from model import QNetwork


def _board_to_numpy(board):
    a1 = np.array([[e['value'] if e is not None else 1 for e in r] for r in board])
    return np.vstack([a1 == 2 ** i for i in range(16)]).reshape(16, 4, 4) * 1

    # return np.array([[e['value'] if e is not None else 0 for e in r] for r in board])


class Env:
    ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT']

    def __init__(self, protocol="http", host="localhost", port=8881):
        self.url = f"{protocol}://{host}:{port}"
        self.json = None

    def reset(self):
        self.json = requests.put(f"{self.url}/reset").json()

    def step(self, action: int):
        self.json = requests.put(f"{self.url}/action", data=Env.ACTIONS[action]).json()

    @property
    def state(self):
        return _board_to_numpy(self.json['board'])

    @property
    def reward(self):
        # Penalize forbidden moves
        if self.invalid:
            return -1
        else:
            return np.log2(self.json['deltaScore'] + 1)

    @property
    def done(self):
        return self.json['done']

    @property
    def score(self):
        return self.json['score']

    @property
    def invalid(self):
        return self.json['invalid']

    @property
    def human(self):
        return self.json['humanBoard']


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
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env.reset()  # reset the environment
        state = env.state  # get the current state
        score = 0
        t = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env.step(action)  # send the action to the environment
            next_state = env.state  # get the next state
            reward = env.reward  # get the reward
            done = env.done
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print(
            '\rEpisode {}\tScore: {}/{:.2f}\tAverage Score: {:.2f}\teps: {:.2f} \t   stp: {}   \t mem: {} '
                .format(i_episode, env.score, score, np.mean(scores_window), eps, t, len(agent.memory)), end="")
        if i_episode % 10 == 0:
            print("")
    return scores


env = Env()
env.reset()
state = env.state

# Create an agent
agent = Agent(QNetwork, len(state), len(Env.ACTIONS), seed=42)

# agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
# agent.qnetwork_target.load_state_dict(torch.load('checkpoint.pth'))

# Execute training
scores = dqn(env, agent, n_episodes=1000, eps_decay=0.999, eps_start=0.05, eps_end=0.001)

# Save trained parameters
torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
