import torch

def cat_padded_seqs(
    a, b, a_lengths, b_lengths, pad_dim=0, batch_dim=1, pad_value=0
):
    device = a.device
    dtype = a.dtype
    
    # compute new lengths
    ab_lengths = a_lengths + b_lengths
    
    # add extra entries required to handle the new maximum length
    max_len = torch.max(ab_lengths)
    extra_len = max_len - a.shape[pad_dim]
    if extra_len > 0:
        extra_values = torch.ones(
            (*a.shape[:pad_dim], extra_len, *a.shape[pad_dim+1:]),
            dtype=dtype,
            device=device,
        ) * pad_value
        ab = torch.cat((a, extra_values), dim=pad_dim)
    else:
        ab = a.clone()
    
    # construct indices for reading and writing elements
    ai = get_pad_range_indices(a_lengths, b_lengths)
    bi, bj = get_pad_batch_indices(b_lengths)
    
    # use pad_dim and batch_dim to pack indices into an index tuple
    #ab_index = [slice(None) for s in ab.shape]
    #ab_index[pad_dim] = ai
    #ab_index[batch_dim] = bj
    ab_index = get_index_tuple(ai, bj, len(ab.shape), pad_dim, batch_dim)
    #b_index = [slice(None) for s in b.shape]
    #b_index[pad_dim] = bi
    #b_index[batch_dim] = bj
    b_index = get_index_tuple(bi, bj, len(b.shape), pad_dim, batch_dim)
    
    # read values from b and write into ab
    ab[tuple(ab_index)] = b[tuple(b_index)]
    
    return ab, ab_lengths

def linearize_padded_seq(x, pad, pad_dim=0, batch_dim=1):
    xi, xj = get_pad_batch_indices(pad)
    #x_index = [slice(None) for s in x.shape]
    #x_index[pad_dim] = xi
    #x_index[batch_dim] = xj
    #return x[tuple(x_index)]
    x_index = get_index_tuple(xi, xj, len(x.shape), pad_dim, batch_dim)
    return x[x_index]

def make_padding_mask(pad, shape, pad_dim=0, batch_dim=1):
    mask = torch.ones(shape, dtype=torch.bool, device=pad.device)
    p_index, b_index = get_pad_batch_indices(pad)
    #index = [slice(None) for s in shape]
    #index[pad_dim] = p_index
    #index[batch_dim] = b_index
    #mask[index] = False
    index = get_index_tuple(p_index, b_index, len(shape), pad_dim, batch_dim)
    mask[index] = False
    
    return mask

def get_pad_batch_indices(pad):
    pad_indices = get_pad_indices(pad)
    batch_indices = get_batch_indices(pad)
    return pad_indices, batch_indices

def get_pad_indices(pad):
    pad_indices = torch.cat([torch.arange(p) for p in pad]).to(pad.device)
    return pad_indices
    
def get_pad_range_indices(pad_start, pad):
    pad_indices = torch.cat([
        torch.arange(ps, ps+p) for ps, p in zip(pad_start, pad)
    ])
    return pad_indices

def get_batch_indices(pad):
    batch_indices = torch.cat([
        torch.ones(p, dtype=torch.long)*i
        for i, p in enumerate(pad)
    ]).to(pad.device)
    return batch_indices

def get_index_tuple(pad_indices, batch_indices, dims, pad_dim, batch_dim):
    index = [slice(None) for d in range(dims)]
    index[pad_dim] = pad_indices
    index[batch_dim] = batch_indices
    return tuple(index)
