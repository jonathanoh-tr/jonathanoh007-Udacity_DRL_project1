import numpy


# SumTree
# a binary tree data structure where the parent�s value is the sum of its children
class SumTree():
    

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0
        self.write=0
        print('tutu')
        
    # update to the root node
    def _propagate(self, idx, change):
        #print('tutu propagate')
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        #print('tutu retrieve')
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        #print('tutu total')
        return self.tree[0]


    # store priority and sample
    def add(self, p, data):
        #print('tutu add')
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        print('write',self.write)
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        #print('tutu update')
        change = p - self.tree[idx]
        #print('tutu update 1')
        #print('tree',self.tree, 'idx',idx, 'p',p)
        #print('self.tree[idx]',self.tree[idx])
        self.tree[idx] = p
        #print('tutu update 2')
        self._propagate(idx, change)
        #print('tutu update 3')
    # get priority and sample
    def get(self, s):
        #print('tutu get')
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])