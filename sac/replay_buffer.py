import collections
import random
import math
import numpy as np


class ReplayBuffer(object):
    def __init__(self, batch_size, max_size):
        self.batch_size = batch_size
        self.max_size = max_size

        self._buffer = collections.deque(maxlen=max_size)

    def add_sample(self, *args):
        for arg in args:
            assert len(arg.shape) == 2
            assert len(arg) == len(args[0])

        for i in range(len(args[0])):
            self._buffer.append([arg[i] for arg in args])

    def random_batch(self):
        n_sample = self.batch_size if len(self._buffer) >= self.batch_size else len(self._buffer)
        t = random.sample(self._buffer, k=n_sample)
        return [np.array(e) for e in zip(*t)]

    def clear(self):
        self._buffer.clear()

    @property
    def is_full(self):
        return len(self._buffer) == self.max_size

    @property
    def size(self):
        return len(self._buffer)

    @property
    def is_lg_batch_size(self):
        return len(self._buffer) > self.batch_size


class PrioritizedReplayBuffer(object):
    def __init__(self, batch_size, max_size, beta=0.9):
        self.batch_size = batch_size
        self.max_size = 2**math.floor(math.log2(max_size))
        self.beta = beta

        self._sum_tree = SumTree(self.max_size)

    def add_sample(self, *args):
        for arg in args:
            assert len(arg.shape) == 2
            assert len(arg) == len(args[0])

        max_weight = self._sum_tree.get_max()

        for i in range(len(args[0])):
            self._sum_tree.add([arg[i] for arg in args], max_weight + 1)

    def random_batch(self):
        n_sample = self.batch_size if self._sum_tree.size >= self.batch_size else self._sum_tree.size
        total = self._sum_tree.get_total()

        step = total // n_sample
        points_transitions_probs = []
        for i in range(n_sample):
            v = np.random.uniform(i * step, (i + 1) * step - 1)
            t = self._sum_tree.sample(v)
            points_transitions_probs.append(t)

        points, transitions, probs = zip(*points_transitions_probs)

        # 计算重要性比率
        importance_ratio = np.array([np.power(self.size * probs[i], -self.beta) for i in range(len(probs))])
        importance_ratio /= importance_ratio.max()

        importance_ratio = np.array(importance_ratio)[:, np.newaxis]

        return points, tuple(np.array(e) for e in zip(*transitions)), importance_ratio

    def update(self, points, td_error):
        for i in range(len(points)):
            self._sum_tree.update(points[i], td_error[i])

    @property
    def size(self):
        return self._sum_tree.size

    @property
    def is_lg_batch_size(self):
        return self._sum_tree.size > self.batch_size


class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)

        self.data = [None] * capacity
        self.size = 0
        self.curr_point = 0

    # 添加一个节点数据
    def add(self, data, weight):
        self.data[self.curr_point] = data

        self.update(self.curr_point, weight)

        self.curr_point += 1
        if self.curr_point >= self.capacity:
            self.curr_point = 0

        if self.size < self.capacity:
            self.size += 1

    # 更新一个节点的优先级权重
    def update(self, point, weight):
        idx = point + self.capacity - 1
        weight += .01
        change = weight - self.tree[idx]

        self.tree[idx] = weight

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def get_total(self):
        return self.tree[0]

    def get_max(self):
        if self.size == 0:
            return 0
        else:
            return max(self.tree[self.capacity - 1:self.capacity + self.size - 1])

    # 根据一个权重进行抽样
    def sample(self, v):
        idx = 0
        while idx < self.capacity - 1:
            l_idx = idx * 2 + 1
            r_idx = l_idx + 1
            if self.tree[l_idx] >= v:
                idx = l_idx
            else:
                idx = r_idx
                v = v - self.tree[l_idx]

        point = idx - (self.capacity - 1)
        # 返回抽样得到的 位置，transition信息，该样本的概率
        return point, self.data[point], self.tree[idx] / self.get_total()


if __name__ == "__main__":
    replay_buffer = PrioritizedReplayBuffer(4, 8)

    replay_buffer.add_sample(np.array([[1], [2], [3]]))
    points, data, importance_ratio = replay_buffer.random_batch()
    print(points, data, importance_ratio)
    replay_buffer.update(points,[1,2,3])
    print(replay_buffer._sum_tree.tree)
    points, data, importance_ratio = replay_buffer.random_batch()
    print(points, data, importance_ratio)
