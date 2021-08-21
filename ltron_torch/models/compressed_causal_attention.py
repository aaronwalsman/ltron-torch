import torch
from torch.nn import Module, Linear, Dropout, ParameterList, ModuleList

from ltron_torch.models.parameter import NoWeightDecayParameter

#def make_compressed_causal_mask(i, temporal_dim=0):
#    i = i[:,:,temporal_dim]
#    n, b = i.shape
#    causal_mask = i.view(n, 1, b) < i.view(1, n, b)
#    return causal_mask

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
    
    def forward(self, x, t, positional_encoding, content_mask, padding_mask):
        '''
        x                  [s, b, c] - data
        t                  [b]       - terminal
        postional_encoding [s, b, c] - 
        content_mask       [s, s, b] - 
        padding_mask       [b, s]    - 
        '''
        s, b, c = x.shape
        
        # compute q,k,v
        qkv = self.qkv_linear(x+positional_encoding)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(s, b, self.h, self.cc)
        k = k.view(s, b, self.h, self.cc)
        v = v.view(s, b, self.h, self.cc)
        
        # add memory to k and v
        if self.training:
            kv_padding_mask = padding_mask
        else:
            k, v, content_mask, kv_padding_mask = self.cat_memory(
                k, v, t, content_mask, padding_mask)
        t = k.shape[0]
        
        # compute attention
        qk = torch.einsum('sbhc,tbhc->stbh', q, k)
        temperature = 1. / self.cc ** 0.5
        a = qk * temperature
        try:
            a.masked_fill(content_mask.view(s, t, b, 1), float('-inf'))
        except:
            import pdb
            pdb.set_trace()
        a.masked_fill(kv_padding_mask.view(1, t, b, 1), float('-inf'))
        p = torch.softmax(a, dim=1)
        p = self.attention_dropout(p)
        
        # compute output content
        x = torch.einsum('stbh,tbhc->sbhc', p, v).reshape(s,b,self.c)
        x = self.content_linear(x)
        x = self.content_dropout(x)
        
        # save memory
        if not self.training:
            self.save_memory(k, v, padding_mask)
        
        return x
    
    def expand_memory_batch(self, b):
        ms, mb = self.memory_k.shape[:2]
        if mb < b:
            new_k = torch.zeros(ms, b-mb, self.h, self.cc)
            new_k = new_k.to(self.memory_k.device)
            self.memory_k = torch.cat((self.memory_k, new_k), dim=1)
            
            new_v = torch.zeros(ms, b-mb, self.h, self.cc)
            new_v = new_v.to(self.memory_v.device)
            self.memory_v = torch.cat((self.memory_v, new_v), dim=1)
            
            new_lengths = torch.zeros(
                b-mb, dtype=torch.long, device=self.memory_length.device)
            self.memory_length = torch.cat((self.memory_length, new_lengths))
    
    def expand_memory_length(self, s):
        ms, mb = self.memory_k.shape[:2]
        if ms < s:
            new_k = torch.zeros(s-ms, mb, self.h, self.cc)
            new_k = new_k.to(self.memory_k.device)
            self.memory_k = torch.cat((self.memory_k, new_k), dim=0)
            
            new_v = torch.zeros(s-ms, mb, self.h, self.cc)
            new_v = new_v.to(self.memory_v.device)
            self.memory_v = torch.cat((self.memory_v, new_v), dim=0)
    
    def cat_memory(self, k, v, t, content_mask, padding_mask):
        s, b, h, c = k.shape
        self.expand_memory_batch(b)
        self.clear_memory(t)
        max_len = torch.max(self.memory_length)
        print('prek', k.shape, self.memory_k.shape)
        print(self.memory_length)
        k = torch.cat((self.memory_k[:max_len], k), dim=0)
        v = torch.cat((self.memory_v[:max_len], v), dim=0)
        print('postk', k.shape)
        
        ms = self.memory_k.shape[0]
        #memory_content_mask = torch.ones(
        #    (ms, s, b), dtype=torch.bool, device=content_mask.device) # s x ms?
        #content_mask = torch.cat((memory_content_mask, content_mask), dim=0)
        memory_content_mask = torch.ones(
            (s, ms, b), dtype=torch.bool, device=content_mask.device)
        content_mask = torch.cat((memory_content_mask, content_mask), dim=1)
        
        memory_padding_mask = torch.zeros((max_len, b), dtype=torch.bool)
        for i in range(b):
            memory_padding_mask[self.memory_length[i]:, i] = True
        memory_padding_mask = memory_padding_mask.to(padding_mask.device)
        kv_padding_mask = torch.cat((memory_padding_mask, padding_mask), dim=0)
        
        return k, v, content_mask, kv_padding_mask
    
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
        kv_i = kv_i[padding_mask]
        memory_i  = memory_i[padding_mask]
        
        # insert into memory
        self.memory_k[memory_i[:,1], memory_i[:,0]] = k[kv_i[:,1], kv_i[:,0]]
        self.memory_v[memory_i[:,1], memory_i[:,0]] = v[kv_i[:,1], kv_i[:,0]]
        
        # update lengths
        self.memory_length = new_lengths
    
    def clear_memory(self, terminal=None):
        if terminal is None:
            self.memory_length = torch.zeros_like(self.memory_length)
        else:
            self.memory_length[terminal] = 0
    
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
        a.masked_fill(cm.unsqueeze(-1), float('-inf'))
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
        p.masked_fill(m, float('-inf'))
        p = torch.softmax(p, dim=1)
        p = self.attention_dropout(p)
        
        x = torch.einsum('stbh,tbhc->sbhc', p, v).view(s,b,self.c)
        x = self.content_linear(x)
        x = self.content_dropout(x)
        
        return x
