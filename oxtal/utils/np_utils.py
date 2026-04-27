import numpy as np


def run_lengths(arr: np.ndarray):
    if arr.size == 0:
        return np.array([], dtype=int)

    # Find where the value changes
    change_indices = np.flatnonzero(arr[1:] != arr[:-1]) + 1
    # Add start and end indices
    indices = np.concatenate(([0], change_indices, [len(arr)]))
    # Compute lengths of runs
    return np.diff(indices)
