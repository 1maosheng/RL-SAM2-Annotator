import random
import numpy as np
from collections import deque


class TrajectoryAwareReplayBuffer:
    def __init__(self, capacity, traj_weights=None):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.traj_labels = {}
        self.traj_weights = traj_weights or {0: 0.1, 1: 0.5, 2: 1.0}
        self.next_traj_id = 0

    def push(self, state, action, reward, next_state, done, traj_id):
        self.buffer.append((state, action, reward, next_state, done, traj_id))

    def set_traj_label(self, traj_id, label):
        mapping = {'invalid': 0, 'suboptimal': 1, 'valid': 2}
        self.traj_labels[traj_id] = mapping[label]

    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        traj_ids = [exp[-1] for exp in self.buffer]
        probs = [self.traj_weights[self.traj_labels[tid]] for tid in traj_ids]
        probs = np.array(probs)
        probs = probs / probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=probs)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)