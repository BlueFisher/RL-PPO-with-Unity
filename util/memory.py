import collections
import random
import numpy as np


class Memory(object):
    def __init__(self, batch_size, max_size):
        self.batch_size = batch_size  # mini batch大小
        self.max_size = max_size
        self.isFull = False
        self._transition_store = collections.deque()

    def store_transition(self, s, a, r, s_, done):
        if len(self._transition_store) == self.max_size:
            self._transition_store.popleft()

        assert len(a.shape) == 1
        self._transition_store.append((s, a, r, s_, done))
        if len(self._transition_store) == self.max_size and not self.isFull:
            self.isFull = True

    def get_mini_batches(self):
        n_sample = self.batch_size if len(self._transition_store) >= self.batch_size else len(self._transition_store)
        t = random.sample(self._transition_store, k=n_sample)
        t = list(zip(*t))

        return tuple(np.array(e) for e in t)

    def clear(self):
        self._transition_store.clear()
        self.isFull = False
