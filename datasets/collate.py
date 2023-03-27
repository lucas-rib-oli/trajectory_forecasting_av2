import torch
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
key_padding = ['lanes']
key_concatenate = ['historic', 'future', 'offset_historic', 'offset_future']

def collate_fn (batch: List[Dict[str, torch.Tensor]]):
    keys = batch[0].keys()
    out = {k: [] for k in keys}
    for data in batch:
        for k, v in data.items():
            out[k].append(v)
    # Padding the lanes
    for k in key_padding:
        out[k] = pad_sequence(out[k], batch_first=True)
    # Concatenate the list of tensors
    for k in key_concatenate:
        out[k] = torch.cat(out[k])
    return out