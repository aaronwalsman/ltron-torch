import torch
from torch.nn import Module, Linear, Dropout, ParameterList, ModuleList

from ltron_torch.models.parameter import NoWeightDecayParameter

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
        
        self.qkv_linear = Linear(self.c, 3*self.c)
        
        memory_k = torch.zeros(
            memory_length, memory_batch_size, self.h, self.cc)
        memory_v = torch.zeros(
            memory_length, memory_batch_size, self.h, self.cc)
        memory_length = torch.zeros((0,), dtype=torch.long)
        self.register_buffer('memory_k', memory_k)#, persistent=False)
        self.register_buffer('memory_v', memory_v)#, persistent=False)
        self.register_buffer('memory_length', memory_length)#, persistent=False)
        
        self.attention_dropout = Dropout(attention_dropout)
        self.content_linear = Linear(self.c, self.c)
        self.content_dropout = Dropout(content_dropout)
    
    def forward(self, x, pe, content_mask, padding_mask, terminal=None):
        '''
        x                  [s, b, c] - data
        t                  [b]       - terminal (used to clear memory)
        pe                 [s, b, c] - positional encoding added to x
        content_mask       [s, s, b] - qk content mask
        padding_mask       [b, s]    - sequence padding mask
        '''
        s, b, c = x.shape
        
        # compute q,k,v
        qkv = self.qkv_linear(x+pe)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(s, b, self.h, self.cc)
        k = k.view(s, b, self.h, self.cc)
        v = v.view(s, b, self.h, self.cc)
        
        # add memory to k and v
        if not self.training:
            k, v, content_mask, padding_mask = self.load_memory(
                k, v, content_mask, padding_mask, terminal)
        t = k.shape[0]
        
        # compute attention
        qk = torch.einsum('sbhc,tbhc->stbh', q, k)
        temperature = 1. / self.cc ** 0.5
        a = qk * temperature
        a = a.masked_fill(content_mask.view(s, t, b, 1), float('-inf'))
        a = a.masked_fill(padding_mask.view(1, t, b, 1), float('-inf'))
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
    
    def load_memory(self, k, v, content_mask, padding_mask, terminal):
        s, b, h, c = k.shape
        
        # expand and clear memory
        self.expand_memory_batch(b)
        self.clear_memory(terminal)
        
        # concatenate memory onto k and v
        max_len = torch.max(self.memory_length)
        mk = torch.cat((self.memory_k[:max_len], k), dim=0)
        mv = torch.cat((self.memory_v[:max_len], v), dim=0)
        
        # compute new content mask
        memory_content_mask = torch.zeros(
            (s, max_len, b), dtype=torch.bool, device=content_mask.device)
        m_content_mask = torch.cat((memory_content_mask, content_mask), dim=1)
        
        # compute new padding mask
        memory_padding_mask = torch.zeros((max_len, b), dtype=torch.bool)
        for i in range(b):
            memory_padding_mask[self.memory_length[i]:, i] = True
        memory_padding_mask = memory_padding_mask.to(padding_mask.device)
        m_padding_mask = torch.cat((memory_padding_mask, padding_mask), dim=0)
        
        # append k and v to memory
        self.save_memory(k, v, padding_mask)
        
        return mk, mv, m_content_mask, m_padding_mask
    
    def save_memory(self, k, v, padding_mask):
        s, b, h, c = k.shape
        
        # expand memory if necessary
        active_elements = ~padding_mask
        kv_lengths = torch.sum(active_elements, dim=0)
        new_lengths = self.memory_length + kv_lengths
        self.expand_memory_length(torch.max(new_lengths))
        
        # generate indices
        kv_i = torch.zeros((s, b, 2), dtype=torch.long, device=k.device)
        kv_i[:,:,0] = torch.arange(s, device=k.device).view(s, 1)
        kv_i[:,:,1] = torch.arange(b, device=k.device).view(1, b)
        memory_i = kv_i.clone()
        memory_i[:,:,0] += self.memory_length.view(1, b)
        kv_ia = kv_i[active_elements]
        memory_i  = memory_i[active_elements]
        
        # insert into memory
        if (torch.any(memory_i[:,0] >= self.memory_k.shape[0]) or
            torch.any(memory_i[:,1] >= self.memory_k.shape[1])):
            print('memory out of bounds')
            import pdb
            pdb.set_trace()
        
        if (torch.any(kv_ia[:,0] >= k.shape[0]) or
            torch.any(kv_ia[:,1] >= k.shape[1])):
            print('k out of bounds')
            import pdb
            pdb.set_trace()
        
        self.memory_k[memory_i[:,0], memory_i[:,1]] = k[kv_ia[:,0], kv_ia[:,1]]
        self.memory_v[memory_i[:,0], memory_i[:,1]] = v[kv_ia[:,0], kv_ia[:,1]]
        
        # update lengths
        self.memory_length = new_lengths
    
    def expand_memory_batch(self, b):
        # if the batch dimension is not large enough, expand
        ms, mb = self.memory_k.shape[:2]
        if mb < b:
            device = self.memory_k.device
            new_k = torch.zeros(ms, b-mb, self.h, self.cc, device=device)
            self.memory_k = torch.cat((self.memory_k, new_k), dim=1)
            
            new_v = torch.zeros(ms, b-mb, self.h, self.cc, device=device)
            self.memory_v = torch.cat((self.memory_v, new_v), dim=1)
            
            new_lengths = torch.zeros(b-mb, dtype=torch.long, device=device)
            self.memory_length = torch.cat((self.memory_length, new_lengths))
    
    def expand_memory_length(self, s):
        # if the sequence dimension is not large enough, expand
        ms, mb = self.memory_k.shape[:2]
        if ms < s:
            device = self.memory_k.device
            new_k = torch.zeros(s-ms, mb, self.h, self.cc, device=device)
            self.memory_k = torch.cat((self.memory_k, new_k), dim=0)
            
            new_v = torch.zeros(s-ms, mb, self.h, self.cc, device=device)
            self.memory_v = torch.cat((self.memory_v, new_v), dim=0)
    
    def clear_memory(self, terminal=None):
        # reset terminal entries to 0, if terminal is None, reset all entries
        if terminal is None:
            self.memory_length = torch.zeros_like(self.memory_length)
        else:
            self.memory_length[terminal.bool()] = 0
    
    def train(self, mode=True):
        super(CompressedCausalAttention, self).train(mode)
        self.clear_memory()


class CompressedRelativeCausalAttention(Module):
    def __init__(self,
        channels,
        heads,
        attention_dropout=0.,
        content_dropout=0.,
    ):
        super(CompressedRelativeCausalAttention, self).__init__()
        assert channels % heads == 0
        self.c = channels
        self.h = heads
        self.cc = self.c // self.h
        
        self.qkv = Linear(self.c, 3*self.c)
        stdv = 1. / channels ** 0.5
        u = torch.zeros(self.c).uniform_(-stdv, stdv)
        w = torch.zeros(self.c).uniform_(-stdv, stdv)
        self.u = NoWeightDecayParameter(u)
        self.w = NoWeightDecayParameter(w)
        
        self.attention_dropout = Dropout(attention_dropout)
        self.content_linear = Linear(self.c, self.c)
        self.content_dropout = Dropout(content_dropout)
    
    def forward(self, x, pe, cm, pm):
        s, b, c = x.shape
        
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(s, b, self.h, self.cc)
        k = k.view(s, b, self.h, self.cc)
        v = v.view(s, b, self.h, self.cc)
        
        # These correspond to the 'u' and 'v' vectors in Transformer-XL,
        # but with 'v' renamed to 'w' to avoid conflict with the v (values)
        # from the qkv notation.
        u = self.u.view(1, 1, self.h, self.cc)
        w = self.w.view(1, 1, self.h, self.cc)
        
        # Following the structure from the Transformer-XL we have four
        # components that sum together to create the attention matrix.
        # In the Transformer-XL these are labelled A, B, C, and D, but here
        # I use the following names:
        # A: dd (data to data)
        # B: dp (data to positional encoding)
        # C: ud (u to data)
        # D: wp (w to positional encoding)
        # Names joined with an underscore indicate a sum:
        # dd_ud = dd+ud (which would be AC in some other implementations).
        
        # Compute dd_ud.  This follows largely from the Transformer-XL
        qu = q + u
        dd_ud = torch.einsum('sbhc,tbhc->stbh', qu, k)
        
        # Compute dp_wp.
        qw = q + w
        pe = pe.view(s, s, b, self.h, self.cc)
        dp_wp = torch.einsum('sbhc,stbhc->stbh', qw, pe)
        
        t = 1. / self.cc ** 0.5
        a = (dd_ud + dp_wp) * t
        #a = dd_ud * t
        a = a.masked_fill(cm.unsqueeze(-1), float('-inf'))
        p = torch.softmax(a, dim=1)
        p = self.attention_dropout(p)
        
        x = torch.einsum('stbh,tbhc->sbhc', p, v).reshape(s,b,self.c)
        x = self.content_linear(x)
        x = self.content_dropout(x)
        
        return x

class SparseRelativeCausalAttention_off(Module):
    def __init__(self,
        channels,
        heads,
        data_shape,
        causal_dims,
        attention_dropout=0.,
        content_dropout=0.,
    ):
        super(SparseRelativeCausalAttention, self).__init__()
        assert channels % heads == 0
        self.c = channels
        self.h = heads
        self.cc = self.c // self.h
        
        self.qkv = Linear(self.c, 3*self.c)
        stdv = 1. / channels ** 0.5
        u = torch.zeros(self.c).uniform_(-stdv, stdv)
        self.u = NoWeightDecayParameter(u)
        w = torch.zeros(self.c).uniform_(-stdv, stdv)
        self.w = NoWeightDecayParameter(w)
        
        # This needs to be moved externally and shared in here among all layers.
        # Also there's a problem with this kind of factorized positional
        # encoding.  You lose some granularity to pay attention to off-axis
        # elements.  You have to say "two steps back in time" and "two steps to
        # the left" which will add weight to EVERYTHING two steps back in time
        # and EVERYTHING two steps to the left.
        self.causal_dims = causal_dims
        #self.embedding_shape = tuple(
        #    d if causal_dim else d*2-1
        #    for d, causal_dim in zip(data_shape, self.causal_dims)
        #)
        #center_offset = torch.LongTensor([
        #    0 if causal_dim else d-1
        #    for d, causal_dim in zip(data_shape, self.causal_dims)
        #])
        #self.register_buffer('center_offset', center_offset)
        #if learned_positional_encoding:
        #    self.positional_encodings = ParameterList([
        #        Parameter(torch.zeros(d, self.c))
        #        for d in self.embedding_shape
        #    ])
        #else:
        #    raise NotImplementedError
        
        self.attention_dropout = Dropout(attention_dropout)
        self.content_linear = Linear(self.c, self.c)
        self.content_dropout = Dropout(content_dropout)
    
    def forward(self, x, e, cm, pm):
        s, b, c = x.shape
        
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(s, b, self.h, self.cc)
        k = k.view(s, b, self.h, self.cc)
        v = v.view(s, b, self.h, self.cc)
        
        # These correspond to the 'u' and 'v' vectors in Transformer-XL,
        # but with 'v' renamed to 'w' to avoid conflict with the v (values)
        # from the qkv notation.
        u = self.u.view(1, 1, self.h, self.cc)
        w = self.w.view(1, 1, self.h, self.cc)
        
        # Following the structure from the Transformer-XL we have four
        # components that sum together to create the attention matrix.
        # In the Transformer-XL these are labelled A, B, C, and D, but here
        # I use the following names:
        # A: ee (embedding to embedding)
        # B: er (embedding to relative encoding)
        # C: ue (u to embedding)
        # D: wr (w to relative encoding)
        # Names joined with an underscore indicate a sum:
        # ee_ue = ee+ue (which would be AC in some other implementations).
        
        # Compute ee_ue.  This follows largely from the Transformer-XL
        qu = q + u
        ee_ue = torch.einsum('sbhc,tbhc->stbh', qu, k)
        
        # Compute er_wr.
        qw = q + w
        
        # Cannot do dot-product on absolute indices and subtract because
        # indices represent discrete vectors.
        # Embedding[i] - Embedding[j] != Embedding[i-j]
        
        # Two options here: factored positional-encoding and non-factored.
        # Factored: do a dot-product between qw and each of the dimensional
        # positional encodings, then select out the appropriate elements and
        # add using the 'i' indices.
        # Non-factored: Use the i-i indices to select out elements of a (very)
        # large positional encoding, then do a dot product between qw and each
        # of those elements.  This seems like it should be about the same
        # ammount of computation as the normal outer-product, but who knows?
        # Let's try the factored version first.
        
        # each element is NxTxBxH where:
        # N: the number of inputs
        # T: the size of this dimension
        # B: batch
        # H: number of heads
        '''
        qwis = [
            torch.einsum('sbhc,thc->stbh', qw, p.view(-1,self.h,self.cc))
            for p in pe.positional_encodings
        ]
        '''
        
        '''
Ok, brainstorm.  I have a list of absolute positions p.  By subtracting p from p^T, I now have a matrix of offsets.  But this is also all batch, so really I have a batch of matrices of offsets.  Now I want to use this to look up a list of positional encodings... why is this so hard?
        '''
        
        '''
        n, b, d = i.shape
        assert n == s
        r = i.view(n, 1, b, d) - i.view(1, n, b, d)
        r = r + self.center_offset
        m = r < 0
        r = torch.clamp(r, min=0)
        '''
        
        
        '''
        er_wr = sum(
            #qwi[range(n),r[:,:,:,j]]
            qwi[
            for j, qwi in enumerate(qwis)
        )
        '''
        t = 1. / self.cc ** 0.5
        p = (ee_ue + er_wr) * t
        p = p.masked_fill(m, float('-inf'))
        p = torch.softmax(p, dim=1)
        p = self.attention_dropout(p)
        
        x = torch.einsum('stbh,tbhc->sbhc', p, v).view(s,b,self.c)
        x = self.content_linear(x)
        x = self.content_dropout(x)
        
        return x
