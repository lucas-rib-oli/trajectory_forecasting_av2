import torch
from typing import List, Dict
from torch.nn.utils.rnn import pad_sequence
key_padding = ['lanes', 'historic', 'future', 'offset_historic', 'offset_future']
key_concatenate = []

def collate_fn (batch: List[Dict[str, torch.Tensor]]):
    keys = batch[0].keys()
    padded_batch = {k: [] for k in keys}
    # Get original lengths
    original_lengths = {k: [] for k in key_padding}
    for key in key_padding:
        original_lengths[key] = [item[key].size(0) for item in batch]
    
    for data in batch:
        for k, v in data.items():
            padded_batch[k].append(v)
    # Padding the lanes
    for k in key_padding:
        padded_batch[k] = pad_sequence(padded_batch[k], batch_first=True)
    # Concatenate the list of tensors
    for k in key_concatenate:
        padded_batch[k] = torch.cat(padded_batch[k])
    
    # Create a mask with the padded values
    padded_lengths = {k: [] for k in key_padding}
    padding_mask = {}
    for key in key_padding:
        padding_mask[key] = torch.zeros(padded_batch[key].shape[0], padded_batch[key].shape[1], dtype=torch.bool)
    # Create mask
    for key in key_padding:
        for i, length in enumerate(original_lengths[key]):
            padding_mask[key][i, length:] = 1
    
    for key in key_padding:
        name_mask_key = key + '_mask'
        padded_batch[name_mask_key] = padding_mask[key]
    return padded_batch