
from collections import deque, namedtuple
import numpy as np
import torch
import random
from Prioritized_memory import Memory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SMALL_EPSILON = 1/1000

class replayMemory():

    def __init__(self, action_size, memory_size, batch_size, seed):
        '''

        :param action_size:
        :param memory_size:
        :param batch_size:
        :param seed:
        '''

        self.action_size = action_size
        self.memory_size=memory_size
        self.memory = Memory(memory_size)

        #self.priority = deque(maxlen=int(memory_size))
        self.small_epsilon = SMALL_EPSILON

        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def __len__(self):

        return self.memory_size

    def sample(self):
        '''

        :return:
        '''

        ''''''
        #TODO 0: np.sum() might be slowing down the code. The code is very slow.
        #priority_weight = self.priority
        #priority_weight = np.asarray(priority_weight, np.float32) / np.asarray(priority_weight, np.float32).sum()
        #index = np.random.choice(np.arange(0, self.__len__()), self.batch_size, replace=False, p=priority_weight)
        #print(priority_weight)
        #print(index)
        #experience = np.random.choice(self.memory, self.batch_size, replace=False, p=priority_weight)

        #experiences = np.asarray(self.memory)[index]


        experiences, index ,priority_weight = self.memory.sample(self.batch_size) 

        ''''''

        #experience = random.sample(self.memory, k=self.batch_size)
        state = torch.from_numpy(np.vstack([exp[0] for exp in experiences if exp is not None])).float().to(device)
        #state = torch.from_numpy(np.array(np.vstack([experience[0]]), dtype=float)).float().to(device)
        action = torch.from_numpy(np.vstack([exp[1] for exp in experiences if exp is not None])).long().to(device)
        reward = torch.from_numpy(np.vstack([exp[2] for exp in experiences if exp is not None])).float().to(device)
        next_state = torch.from_numpy(np.vstack([exp[3] for exp in experiences if exp is not None])).float().to(device)
        
        done =torch.from_numpy(np.vstack([exp[4] for exp in experiences if exp is not None]).astype(np.uint8)).float().to(device)
        return (state, action, reward, next_state, done, index)


    def add_priority(self, delta, index):

        '''updating priority weights'''
        print(index)
        self.memory.update(index,delta)


    def add(self, state, action, reward, next_state, done):
        experience = self.experience(state, action, reward, next_state, done)
        self.memory.add(self.small_epsilon,experience)
    '''
    call a function that gives back an index of size batch_size and se that to sample from emory buffer using np.random.choice
    
    need a vector pw ith priorty weight
    
    '''