print("****************************")
print("Loading Double Q Learning Memory")
print("****************************")

from collections import deque, namedtuple
import numpy as np
import torch
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class replayMemory():

    def __init__(self, action_size, memory_size, batch_size, seed):
        '''
        :param action_size:
        :param memory_size:
        :param batch_size:
        :param seed:
        '''

        self.action_size = action_size
        self.memory = deque(maxlen=int(memory_size))
        self.priority = deque()
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def __len__(self):

        return len(self.memory)

    def sample(self):
        '''
        :return:
        '''

        experience = random.sample(self.memory, k=self.batch_size)

        state = torch.from_numpy(np.vstack([exp.state for exp in experience if exp is not None])).float().to(device)
        action = torch.from_numpy(np.vstack([exp.action for exp in experience if exp is not None])).long().to(device)
        reward = torch.from_numpy(np.vstack([exp.reward for exp in experience if exp is not None])).float().to(device)
        next_state = torch.from_numpy(np.vstack([exp.next_state for exp in experience if exp is not None])).float().to \
            (device)
        done = torch.from_numpy \
            (np.vstack([exp.done for exp in experience if exp is not None]).astype(np.uint8)).float().to(device)

        return (state, action, reward, next_state, done)

    def add(self, state, action, reward, next_state, done):

        experience = self.experience(state, action, reward, next_state, done)
        self.memory.append(experience)

    '''
    call a function that gives back an index of size batch_size and se that to sample from emory buffer using np.random.choice

    need a vector pw ith priorty weight

    '''