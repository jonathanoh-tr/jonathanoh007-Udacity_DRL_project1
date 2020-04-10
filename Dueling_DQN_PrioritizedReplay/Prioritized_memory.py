import random
import numpy as np
from SumTree import SumTree

class Memory():  # stored as ( s, a, r, s_ ) in SumTree

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.e = 0.01
        self.a = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.001
        
        
    def _get_priority(self, error):
        #print('titi get priority')
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, sample):
        #print('titi add')
        p = self._get_priority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        #print('titi sample')
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            ac = segment * i
            bc = segment * (i + 1)

            s = random.uniform(ac, bc)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    def update(self, idx, error):
        #print('titi update')
        p = self._get_priority(error)
        self.tree.update(idx, p)