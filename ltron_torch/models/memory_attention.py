import torch
from torch.nn import Module, Linear, Dropout, ParameterList, ModuleList

from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.models.padding import cat_padded_seqs, make_padding_mask

class MemoryAttention(Module):
    '''
    Implements a standard multi-head attention module, but adds memory
    functionality to enable more efficient step-by-step rollouts.  To do this
    this module stores past k and v values in the memory_kv buffer as well as
    the length of each sequence in the current batch in the memory_length
    buffer.
    
    The primary interface to using memory is the "use_memory" argument of the
    "forward" method.  If this argument is None, the memory will not be used or
    modified at all, and this module will act like a normal MultiHeadAttention.
    If the argument is a vector of booleans, the k and v values stored in
    memory will be prepended to the k and v values computed during the forward
    call, and the new k and v values will be appended to the memory after the
    forward call for all non-zero entries in the vector.  For all zero entries,
    the memory will be reset to zero length.
    
    The memory will also be reset every time the "train" method is called.
    '''
    def __init__(self,
        channels,
        heads,
        attention_dropout=0.,
        content_dropout=0.,
    ):
        super(MemoryAttention, self).__init__()
        
        # store values we need for later
        assert channels % heads == 0
        self.c = channels
        self.h = heads
        self.hc = self.c // self.h
        
        # register memory buffers
        self.register_buffer('memory_kv', torch.zeros(0, 0, self.h, 2*self.hc))
        self.register_buffer(
            'memory_length', torch.zeros((0,), dtype=torch.long))
        
        # build qkv layers
        self.qkv_linear = Linear(self.c, 3*self.c)
        self.attention_dropout = Dropout(attention_dropout)
        
        # build forward projection
        self.content_linear = Linear(self.c, self.c)
        self.content_dropout = Dropout(content_dropout)
    
    def forward(self,
        xq, pad_q, xk=None, pad_k=None, mask=None, use_memory=None,
    ):
        # set defaults
        if xk is None:
            xk = xq
        if pad_k is None:
            pad_k = pad_q
        
        # pull shape
        s, b, c = x.shape
        
        # compute q,k,v
        qkv = self.qkv_linear(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(s, b, self.h, self.hc)
        k = k.view(s, b, self.h, self.hc)
        v = v.view(s, b, self.h, self.hc)
        
        # add memory to k and v if in eval mode
        if use_memory is not None:
            k, v, mask = self.load_memory(k, v, mask, pad, use_memory)
        t = k.shape[0]
        
        # compute attention
        qk = torch.einsum('sbhc,tbhc->stbh', q, k)
        temperature = 1. / self.hc ** 0.5
        a = qk * temperature
        if mask is not None:
            a = a.masked_fill(mask.view(s, t, b, 1), float('-inf'))
        p = torch.softmax(a, dim=1)
        p = torch.nan_to_num(p)
        p = self.attention_dropout(p)
        
        # compute forward projection
        x = torch.einsum('stbh,tbhc->sbhc', p, v).reshape(s,b,self.c)
        x = self.content_linear(x)
        x = self.content_dropout(x)
        
        return x
    
    def load_memory(self, k, v, mask, pad, use_memory):
        s, b, h, c = k.shape
        
        # expand and clear memory
        self.expand_memory_batchsize(b)
        self.clear_memory_entries(~use_memory.bool())
        
        # compute new mask
        if mask is not None:
            expanded_mask = make_padding_mask(
                self.memory_length, (s, torch.max(self.memory_length), b),
                seq_dim=1, batch_dim=2)
            expanded_mask, _ = cat_padded_seqs(
                expanded_mask, mask, self.memory_length, pad,
                seq_dim=1, batch_dim=2, pad_value=True)
            mask = expanded_mask
        
        # concatenate k and v onto the memory
        kv = torch.cat((k, v), dim=-1)
        self.memory_kv, self.memory_length = cat_padded_seqs(
            self.memory_kv, kv, self.memory_length, pad)
        
        # chop off extra sequence elements
        max_len = torch.max(self.memory_length)
        mkv = self.memory_kv[:max_len]
        mk = mkv[...,:self.hc]
        mv = mkv[...,self.hc:]
        
        return mk, mv, mask
    
    def expand_memory_batchsize(self, b):
        # if the batch dimension is not large enough, expand
        ms, mb = self.memory_kv.shape[:2]
        if mb < b:
            device = self.memory_kv.device
            new_kv = torch.zeros(ms, b-mb, self.h, self.hc*2, device=device)
            self.memory_kv = torch.cat((self.memory_kv, new_kv), dim=1)
            
            new_lengths = torch.zeros(b-mb, dtype=torch.long, device=device)
            self.memory_length = torch.cat((self.memory_length, new_lengths))
    
    def clear_memory_entries(self, clear_entries):
        self.memory_length[clear_entries] = 0
    
    def clear_all_memory(self):
        self.memory_length.fill_(0)
    
    def train(self, mode=True):
        super(MemoryAttention, self).train(mode)
        self.clear_all_memory()
