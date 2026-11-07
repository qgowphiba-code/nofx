"""Import-name shim for the SMPL benchmark's helper package (mzutils), whose real distribution pulls in torchvision,
sklearn and nltk. Only the names smpl.envs imports are provided; normalize/denormalize follow mzutils' [-1, 1] convention.
The verification runs call the environments with normalize=False, so these two functions are not on that path."""
import heapq, os
import numpy as np


class SimplePriorityQueue:
    def __init__(self):
        self.h = []

    def push(self, item, priority):
        heapq.heappush(self.h, (priority, item))

    def pop(self):
        return heapq.heappop(self.h)[1]

    def __len__(self):
        return len(self.h)


def normalize_spaces(x, max_x, min_x):
    x = np.asarray(x, dtype=np.float32); max_x = np.asarray(max_x, dtype=np.float32); min_x = np.asarray(min_x, dtype=np.float32)
    return (x - min_x) / (max_x - min_x) * 2.0 - 1.0, max_x, min_x


def denormalize_spaces(x, max_x, min_x):
    x = np.asarray(x, dtype=np.float32); max_x = np.asarray(max_x, dtype=np.float32); min_x = np.asarray(min_x, dtype=np.float32)
    return (x + 1.0) / 2.0 * (max_x - min_x) + min_x, max_x, min_x


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def get_things_in_loc(path, just_files=True):
    if not os.path.isdir(path):
        return []
    return [os.path.join(path, f) for f in sorted(os.listdir(path))]
