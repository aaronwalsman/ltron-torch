import torch
from torch.nn import Module, Linear, Dropout, ParameterList, ModuleList

from ltron_torch.models.parameter import NoWeightDecayParameter
from ltron_torch.models.padding import cat_padded_seqs

class CompressedCausalAttention(Module):
    def __init__(self,
        channels,
        heads,
        attention_dropout=0.,
        content_dropout=0.,
        memory_length=0,
        memory_batch_size=0,
    ):
        super(CompressedCausalAttention, self).__init__()
        assert channels % heads == 0
        self.c = channels
        self.h = heads
        self.cc = self.c // self.h
        
        # qkv layer
        self.qkv_linear = Linear(self.c, 3*self.c)
        
        # memory
        memory_kv = torch.zeros(
            memory_length, memory_batch_size, self.h, 2*self.cc)
        memory_length = torch.zeros((0,), dtype=torch.long)
        self.register_buffer('memory_kv', memory_kv)#, persistent=False)
        self.register_buffer('memory_length', memory_length)#, persistent=False)
        
        # dropout
        self.attention_dropout = Dropout(attention_dropout)
        self.content_linear = Linear(self.c, self.c)
        self.content_dropout = Dropout(content_dropout)
    
    def forward(self, x, pe, content_mask, pad, terminal=None):
        s, b, c = x.shape
        
        # compute q,k,v
        qkv = self.qkv_linear(x+pe)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(s, b, self.h, self.cc)
        k = k.view(s, b, self.h, self.cc)
        v = v.view(s, b, self.h, self.cc)
        
        # add memory to k and v
        if not self.training:
            k, v, content_mask = self.load_memory(
                k, v, content_mask, pad, terminal)
        t = k.shape[0]
        
        # compute attention
        qk = torch.einsum('sbhc,tbhc->stbh', q, k)
        temperature = 1. / self.cc ** 0.5
        a = qk * temperature
        a = a.masked_fill(content_mask.view(s, t, b, 1), float('-inf'))
        p = torch.softmax(a, dim=1)
        p = self.attention_dropout(p)
        
        if torch.any(torch.isnan(p)):
            print('nan in p')
            import pdb
            pdb.set_trace()
        
        # compute output content
        x = torch.einsum('stbh,tbhc->sbhc', p, v).reshape(s,b,self.c)
        x = self.content_linear(x)
        x = self.content_dropout(x)
        
        return x
    
    def load_memory(self, k, v, content_mask, pad, terminal):
        s, b, h, c = k.shape
        
        # expand and clear memory
        self.expand_memory_batch(b)
        self.clear_memory(terminal)
        
        # compute new content_mask
        m_content_mask = torch.zeros(
            (s, torch.max(self.memory_length), b),
            dtype=torch.bool,
            device=content_mask.device,
        )
        m_content_mask, _ = cat_padded_seqs(
            m_content_mask, content_mask, self.memory_length, pad,
            pad_dim=1, batch_dim=2, pad_value=True)
        
        # concatenate memory onto k and v
        kv = torch.cat((k, v), dim=-1)
        self.memory_kv, self.memory_length = cat_padded_seqs(
            self.memory_kv, kv, self.memory_length, pad)
        
        # chop off any extra bits from long unused memory storage
        max_len = torch.max(self.memory_length)
        mkv = self.memory_kv[:max_len]
        mk = mkv[...,:self.cc]
        mv = mkv[...,self.cc:]
        
        return mk, mv, m_content_mask
    
    def expand_memory_batch(self, b):
        # if the batch dimension is not large enough, expand
        ms, mb = self.memory_kv.shape[:2]
        if mb < b:
            device = self.memory_kv.device
            new_kv = torch.zeros(ms, b-mb, self.h, self.cc*2, device=device)
            self.memory_kv = torch.cat((self.memory_kv, new_kv), dim=1)
            
            new_lengths = torch.zeros(b-mb, dtype=torch.long, device=device)
            self.memory_length = torch.cat((self.memory_length, new_lengths))
    
    def clear_memory(self, terminal=None):
        # reset terminal entries to 0, if terminal is None, reset all entries
        if terminal is None:
            self.memory_length = torch.zeros_like(self.memory_length)
        else:
            self.memory_length[terminal.bool()] = 0
    
    def train(self, mode=True):
        super(CompressedCausalAttention, self).train(mode)
        self.clear_memory()
