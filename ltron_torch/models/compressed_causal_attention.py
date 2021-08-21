import torch
from torch.nn import Module, Linear, Dropout, ParameterList, ModuleList

from ltron_torch.models.positional_encoding import positional_encoding
from ltron_torch.models.parameter import NoWeightDecayParameter

class NDLearnedRelativePositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims):
        super(NDLearnedPositionalEncoding, self).__init__()
        encoding_shape = tuple(
            d if (i in causal_dims) else d*2-1
            for i, d in enumerate(data_shape)
        )
        self.register_buffer('center_offset', torch.LongTensor([
            0 if (i in causal_dims) else d-1
            for i,d in enumerate(data_shape)
        ]))
        self.dim_encodings = ParameterList([
            NoWeightDecayParameter(torch.zeros(d, channels))
            for d in encoding_shape
        ])
    
    def forward(self, i):
        n, b, d = i.shape
        r = i.view(n,1,b,d) - i.view(1,n,b,d)
        r = r + self.center_offset.view(1,1,1,d)
        cm = torch.max(r < 0, dim=-1)[0]
        r = torch.clamp(r, min=0)
        
        pe = sum(p[r[:,:,:,j]] for j,p in enumerate(self.dim_encodings))
        
        return pe, cm

class NDLearnedPositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims):
        super(NDLearnedPositionalEncoding, self).__init__()
        self.dim_encodings = ParameterList([
            NoWeightDecayParameter(torch.zeros(d, channels))
            for d in data_shape
        ])
        self.causal_dims = causal_dims
    
    def forward(self, i):
        n, b, d = i.shape
        #r = i.view(n,1,b,d) - i.view(1,n,b,d)
        #r = r + self.center_offset.view(1,1,1,d)
        #cm = torch.max(r < 0, dim=-1)[0]
        #r = torch.clamp(r, min=0)
        
        #pe = sum(p[r[:,:,:,j]] for j,p in enumerate(self.dim_encodings))
        
        pe = sum(p[i[:,:,j]] for j,p in enumerate(self.dim_encodings))
        
        # TMP
        cm = torch.zeros((n, n, b), dtype=torch.bool, device=i.device)
        
        return pe, cm

class NDPositionalEncoding(Module):
    def __init__(self, channels, data_shape, causal_dims):
        super(NDPositionalEncoding, self).__init__()
        positional_linears = []
        for i, d in enumerate(data_shape):
            self.register_buffer('pe_%i'%i, positional_encoding(channels, d))
            positional_linears.append(Linear(channels, channels))
        self.positional_linears = ModuleList(positional_linears)
        
        self.causal_dims = causal_dims
        self.d = len(data_shape)
    
    def forward(self, i):
        n, b, d = i.shape
        pe = sum(
            self.positional_linears[j](getattr(self, 'pe_%i'%j)[i[:,:,j]])
            for j in range(self.d)
        )
        
        # TMP
        cm = torch.zeros((n, n, b), dtype=torch.bool, device=i.device)
        
        return pe, cm

class CompressedCausalAttention(Module):
    def __init__(self,
        channels,
        heads,
        attention_dropout=0.,
        content_dropout=0.,
    ):
        super(CompressedCausalAttention, self).__init__()
        assert channels % heads == 0
        self.c = channels
        self.h = heads
        self.cc = self.c // self.h
        
        self.qkv = Linear(self.c, 3*self.c)
        
        self.attention_dropout = Dropout(attention_dropout)
        self.content_linear = Linear(self.c, self.c)
        self.content_dropout = Dropout(content_dropout)
    
    def forward(self, x, pe, cm, pm):
        s, b, c = x.shape
        
        qkv = self.qkv(x+pe)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = q.view(s, b, self.h, self.cc)
        k = k.view(s, b, self.h, self.cc)
        v = v.view(s, b, self.h, self.cc)
        
        qk = torch.einsum('sbhc,tbhc->stbh', q, k)
        t = 1. / self.cc ** 0.5
        a = qk * t
        a.masked_fill(cm.unsqueeze(-1), float('-inf'))
        p = torch.softmax(a, dim=1)
        p = self.attention_dropout(p)
        
        x = torch.einsum('stbh,tbhc->sbhc', p, v).reshape(s,b,self.c)
        x = self.content_linear(x)
        x = self.content_dropout(x)
        
        return x


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
        #qw = q + w
        #pe = pe.view(s, s, b, self.h, self.cc)
        #dp_wp = torch.einsum('sbhc,stbhc->stbh', qw, pe)
        
        t = 1. / self.cc ** 0.5
        #a = (dd_ud + dp_wp) * t
        a = dd_ud * t
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
