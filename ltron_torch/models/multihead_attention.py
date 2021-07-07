import torch
import torch.nn.functional as F
from torch.nn import Module

import ltron_torch.models.transformer_masks as transformer_masks

def sparse_multihead_attention(
    q, k, v, n_head, q_id, k_id, dropout=0, training=False
):
    s, b, c = v.shape
    assert c % n_head == 0
    cc = c // n_head
    q = q.view(s, b, n_head, cc)
    k = k.view(s, b, n_head, cc)
    v = v.view(s, b, n_head, cc)
    qk = torch.einsum('sbhc,sbhc->sbh', q[q_id], k[k_id]).unsqueeze(-1)
    qk = qk / (cc**0.5)
    
    # limited-stability softmax ------------------------------------------------
    # The following implements a parallel variable-length softmax due to the
    # potentially varying number of attention-comparisons per entry.  When
    # computing a softmax, it is best to subtract the max entry in order to
    # reduce numerical issues.  Instead of doing that independently for each
    # softmax, I take the max over the entire set of parallel softmaxes because
    # it's easier than computing a max for each, and it's better than nothing.
    max_qk = torch.max(qk)
    eqk = torch.exp(qk - max_qk)
    d = torch.zeros(s, b, n_head, 1)
    d.index_add_(0, q_id, eqk)
    
    eqk = F.dropout(eqk, dropout, training=training)
    qkv = eqk * v[k_id]
    x = torch.zeros(s, b, n_head, cc)
    x.index_add_(0, q_id, qkv)
    x = x / d
    
    x = x.view(s, b, c)
    
    return x

class SparseMultiheadAttention(Module):
    def __init__(
        self, n_embed, n_head, q_id, k_id,
        attention_dropout=0.,
        residual_dropout=0.,
    ):
        self.q_linear = Linear(n_embed, n_embed)
        self.k_linear = Linear(n_embed, n_embed)
        self.v_linear = Linear(n_embed, n_embed)
        
        self.register_buffer('q_id', q_id)
        self.register_buffer('k_id', k_id)
        
        self.attention_dropout = attention_dropout
        
        self.x_linear = Linear(n_embed, n_embed)
        self.x_dropout = Dropout(residual_dropout)
        
        self.n_head = n_head
    
    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        
        x = sparse_multihead_attention(
            q, k, v, self.n_head, self.q_id, self.k_id,
            dropout=self.attention_dropout, training=self.training)
        x = self.x_linear(x)
        x = self.x_dropout(x)
        
        return x

# minGPT Multihead Attention ===================================================
# The following functions have been derived from Karpathy's minGPT example at:
# https://github.com/karpathy/minGPT/blob/master/mingpt/model.py
# They are not verbatim copies, but have been modified somewhat to match the
# structure of the rest of the code.

def mingpt_multihead_attention(
    q, k, v, n_head, mask,
    dropout=0, training=False
):
    T,B,C = q.shape
    q = q.permute(1,0,2)
    k = k.permute(1,0,2)
    v = v.permute(1,0,2)
    
    h,w = mask.shape
    mask = mask.view(1,1,h,w)
    
    # calculate query, key, values for all heads in batch
    # and move head forward to be the batch dim
    # (B, nh, T, hs)
    k = k.view(B, T, n_head, C // n_head).transpose(1, 2)
    # (B, nh, T, hs)
    q = q.view(B, T, n_head, C // n_head).transpose(1, 2)
    # (B, nh, T, hs)
    v = v.view(B, T, n_head, C // n_head).transpose(1, 2)
    
    # causal self-attention;
    # Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
    att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)**0.5)
    att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
    att = F.softmax(att, dim=-1)
    
    #att = self.attn_drop(att)
    att = F.dropout(att, dropout, training=training)
    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    # re-assemble all head outputs side by side
    y = y.transpose(1, 2).contiguous().view(B, T, C)
    # output projection
    #y = self.resid_drop(self.proj(y))
    return y.permute(1,0,2)
    '''

class MinGPTMultiheadAttention(Module):
    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, mask):
        super(MinGPTMultiheadAttention, self).__init__()
        assert channels % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left
        # in the input sequence
        self.register_buffer("mask", mask)
        self.n_head = n_head

def test():
    k = torch.rand(16,4,256) * 2. - 1.
    q = torch.rand(16,4,256) * 2. - 1.
    v = torch.rand(16,4,256) * 2. - 1.
    
    mask = ~transformer_masks.jellyfish_mask(4, 3)
    #mask = torch.ones((3,3), dtype=torch.bool)
    
    x1 = sparse_multihead_attention(q, k, v, 4, mask, 0., False)
    x2 = karpathy_attn(q, k, v, 4, mask, 0., False)
    
    print(torch.allclose(x1,x2, rtol=1e-3))   

if __name__ == '__main__':
    test()
'''
