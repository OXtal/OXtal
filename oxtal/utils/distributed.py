import os
import pickle

import pandas as pd
import torch


class DistWrapper:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)
        # self.global_rank =


DIST_WRAPPER = DistWrapper()


def traverse_and_aggregate(gathered_dict, aggregation_func=None):
    """Traverse list of dicts and merge into a single dict with leaf values joined to list."""
    merged_dict = {}
    keys = gathered_dict[0].keys()
    for key in keys:
        value = [d[key] for d in gathered_dict if key in d]
        if isinstance(value[0], dict):
            merged_dict[key] = traverse_and_aggregate(value, aggregation_func=aggregation_func)
        else:
            if aggregation_func is not None:
                value = aggregation_func(value)
            merged_dict[key] = value

    return merged_dict


def gather_and_merge(fabric, metrics, aggregation_func=None):
    """Gather metrics from ddp workers and aggregate leaf metrics."""
    # print(metrics)
    # fix different size
    fixed_metrics = fabric.all_gather(metrics)
    merged_metrics = traverse_and_aggregate([fixed_metrics], aggregation_func)
    return merged_metrics


def gather_dfs(fabric, df):
    serialized = pickle.dumps(df)

    # Convert to tensor
    byte_tensor = torch.ByteTensor(list(serialized))

    # All gather lengths first
    length = torch.tensor([len(byte_tensor)], device=fabric.device)
    all_lengths = fabric.all_gather(length)
    max_len = all_lengths.max().item()

    # Pad tensors to max_len
    if len(byte_tensor) < max_len:
        pad = torch.zeros(max_len - len(byte_tensor), dtype=torch.uint8)
        byte_tensor = torch.cat([byte_tensor, pad])

    gathered_bytes = fabric.all_gather(byte_tensor)

    # Deserialize
    dfs = []
    for i, length in enumerate(all_lengths):
        serialized = bytes(gathered_bytes[i][:length])
        dfs.append(pickle.loads(serialized))

    return pd.concat(dfs)
