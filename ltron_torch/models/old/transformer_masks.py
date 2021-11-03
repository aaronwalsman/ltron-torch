import math

import torch
import torch.nn.functional as F

def full(n):
    return torch.zeros((n, n), dtype=torch.bool)

def neighborhood(h, w, width=1):
    mask = torch.ones((h,w,h,w), dtype=torch.bool)
    for y in range(h):
        for yy in range(y-width, y+width+1):
            if yy < 0 or yy >= h:
                continue
            for x in range(w):
                for xx in range(x-width, x+width+1):
                    if xx < 0 or xx >= w:
                        continue
                    
                    mask[y,x,yy,xx] = False
    
    mask = mask.reshape(h*w, h*w)
    
    return mask

def causal(n):
    return torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)

def blowfish(h, n, s):
    '''
    h : hidden tokens
    n : spatial tokens
    s : sequence length
    '''
    num_tokens = h + n * s
    mask = ~torch.eye(num_tokens, dtype=torch.bool)
    mask[:h] = False
    mask[:,:h] = False

    return mask

def octopus(h, n, s):
    '''
    h : hidden tokens
    n : spatial tokens
    s : sequence length
    '''
    mask = blowfish(h, n, s)
    block = ~torch.eye(n, dtype=torch.bool)
    for i in range(s):
        for j in range(s):
            if i == j:
                continue
            mask[h+i*n:h+(i+1)*n, h+j*n:h+(j+1)*n] = block

    return mask

def anemone(n, s):
    num_tokens = n * (1+s)
    mask = ~torch.eye(num_tokens, dtype=torch.bool)
    mask[:n, :n] = False
    block = ~torch.eye(n, dtype=torch.bool)
    for l in range(1, s+1):
        mask[:n, l*n:(l+1)*n] = block
        mask[l*n:(l+1)*n, :n] = block

    return mask

def jellyfish(n, s):
    '''
    One head bundle of tokens that can all talk to each other
    Each leg grows out of one head token and talk to the head token
    '''
    mask = anenome(n, s)
    block = ~torch.eye(n, dtype=torch.bool)
    for i in range(1, s+1):
        for j in range(1, s+1):
            if i == j:
                continue
            mask[i*n:(i+1)*n, j*n:(j+1)*n] = block

    return mask
