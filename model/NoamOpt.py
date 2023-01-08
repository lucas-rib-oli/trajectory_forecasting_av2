import torch
import numpy as np

class NoamOpt(object):
    """
    Optim wrapper that implements rate.
    """
    def __init__(self, model_size, warmup, optimizer, factor = 1.0):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        self.epoch=0
        self.best_mad=np.inf
        self.best_fad=np.inf

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement 'rate' above
        """
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** -0.5) * min(step ** -0.5, step * self.warmup ** -1.5)


def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
