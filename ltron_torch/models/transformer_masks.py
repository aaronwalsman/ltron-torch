import math

import torch
import torch.nn.functional as F

def stingray(n):
    return torch.tril(torch.ones(n, n, dtype=torch.bool), diagonal=-1)

def neighbor(h, w, width=1):
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

def anenome(n, s):
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

# adapted from Karpathy mingpt for reference
def karpathy_attn(q,k,v,n_head, mask):
        
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
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        #att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # output projection
        #y = self.resid_drop(self.proj(y))
        return y.permute(1,0,2)

def sparse_multihead_attention(q, k, v, n_heads, mask):
    
    #qk^T/d**0.5
    s, b, c = v.shape
    cc = c // n_heads
    q = q.view(s, b, n_heads, cc)
    k = k.view(s, b, n_heads, cc)
    v = v.view(s, b, n_heads, cc)
    ki, qi = torch.nonzero(mask, as_tuple=True)
    qk = torch.einsum('sbhc,sbhc->sbh', q[qi], k[ki]).unsqueeze(-1)
    qk = qk / (cc**0.5)
    
    # limited-stability softmax
    # rather than stabilizing using the max of each sum, we stabilize with
    # the max over all sums
    max_qk = torch.max(qk)
    eqk = torch.exp(qk - max_qk)
    d = torch.zeros(s, b, n_heads, 1)
    d.index_add_(0, qi, eqk)
    
    qkv = eqk * v[ki]
    y = torch.zeros(s, b, n_heads, cc)
    y.index_add_(0, qi, qkv)
    y = y / d
    
    y = y.view(s, b, c)
    
    return y

def test():
    k = torch.rand(16,4,256) * 2. - 1.
    q = torch.rand(16,4,256) * 2. - 1.
    v = torch.rand(16,4,256) * 2. - 1.
    
    mask = ~jellyfish(4, 3)
    #mask = torch.ones((3,3), dtype=torch.bool)
    
    x1 = sparse_multihead_attention(q, k, v, 4, mask)
    x2 = karpathy_attn(q, k, v, 4, mask)
    
    print(torch.allclose(x1,x2, rtol=1e-3))   

if __name__ == '__main__':
    test()
