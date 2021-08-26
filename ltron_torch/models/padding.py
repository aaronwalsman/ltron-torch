import torch

def cat_padded_seqs(
    a, b, a_lengths, b_lengths, pad_dim=0, batch_dim=1, pad_value=0
):
    device = a.device
    dtype = a.dtype
    ab_lengths = a_lengths + b_lengths
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
    
    ai = torch.cat([
        torch.arange(la,la+lb, device=device)
        for la, lb in zip(a_lengths, b_lengths)
    ])
    bi = torch.cat([torch.arange(l, device=device) for l in b_lengths])
    bj = torch.cat([
        torch.ones(l, dtype=torch.long, device=device)*i
        for i, l in enumerate(b_lengths)
    ])
    ab_index = [slice(None) for s in ab.shape]
    ab_index[pad_dim] = ai
    ab_index[batch_dim] = bj
    b_index = [slice(None) for s in b.shape]
    b_index[pad_dim] = bi
    b_index[batch_dim] = bj
    ab[tuple(ab_index)] = b[tuple(b_index)]
    
    return ab, ab_lengths

def linearize_padded_seq(x, lengths, pad_dim=0, batch_dim=1):
    device = x.device
    xi = torch.cat([torch.arange(l, device=device) for l in lengths])
    xj = torch.cat([
        torch.ones(l, dtype=torch.long, device=device)*i
        for i, l in enumerate(lengths)
    ])
    x_index = [slice(None) for s in x.shape]
    x_index[pad_dim] = xi
    x_index[batch_dim] = xj
    return x[tuple(x_index)]

def make_padding_mask(lengths, shape, pad_dim=0, batch_dim=1):
    device = lengths.device
    mask = torch.ones(shape, dtype=torch.bool, device=device)
    p_index = torch.cat([torch.arange(l, device=device) for l in lengths])
    b_index = torch.cat([
        torch.ones(l, dtype=torch.long, device=device)*i
        for i, l in enumerate(lengths)
    ])
    index = [slice(None) for s in shape]
    index[pad_dim] = p_index
    index[batch_dim] = b_index
    mask[index] = False
    
    return mask
