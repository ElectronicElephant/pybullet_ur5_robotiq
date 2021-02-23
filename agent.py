from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import namedtuple


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state'))


class BaseAgent(object):
    def act(self, observation, reward, done):
        pass


class RandomAgent(BaseAgent):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()
