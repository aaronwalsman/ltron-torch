import torch
from torch.distributions import Bernoulli, Categorical

def bernoulli_or_max(logits, mode):
    if mode == 'sample':
        distribution = Bernoulli(logits=logits)
        return distribution.sample()
    elif mode == 'max':
        return (logits > 0.).long()
    else:
        raise NotImplementedError

def categorical_or_max(logits, mode):
    if mode == 'sample':
        distribution = Categorical(logits=logits)
        return distribution.sample()
    elif mode == 'max':
        return torch.argmax(logits, dim=-1)
    else:
        raise NotImplementedError

def categorical_or_max_2d(logits, mode):
    *dims, h, w = logits.shape
    logits = logits.view(*dims, h*w)
    yx = categorical_or_max(logits, mode)
    y = torch.div(yx, w, rounding_mode='floor')
    x = yx % w
    return y, x

def categorical_or_max_nd(logits, mode, n):
    dims = logits.shape[:-n]
    n_dims = logits.shape[-n:]
    logits = logits.reshape(*dims, -1)
    k = categorical_or_max(logits, mode)
    out = []
    for n_dim in reversed(n_dims):
        out.append(k % n_dim)
        k = torch.div(k, n_dim, rounding_mode='floor')
    return tuple(reversed(out))
