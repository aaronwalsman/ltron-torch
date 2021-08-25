import torch

def cat_padded_sequences(a, b, a_lengths, b_lengths):
    device = a.device
    dtype = a.dtype
    ab_lengths = a_lengths + b_lengths
    max_len = torch.max(ab_lengths)
    extra_len = max_len - a.shape[0]
    extra_zeros = torch.zeros(
        (extra_len, *a.shape[1:]),
        dtype=dtype,
        device=device,
    )
    ab = torch.cat((a, extra_zeros), dim=0)
    ai = torch.cat([
        torch.arange(la,la+lb, device=device)
        for la, lb in zip(a_lengths, b_lengths)
    ])
    bi = torch.cat([torch.arange(l, device=device) for l in b_lengths])
    bj = torch.cat([
        torch.ones(l, dtype=torch.long, device=device)*i
        for i, l in enumerate(b_lengths)
    ])
    ab[ai, bj] = b[bi, bj]
    
    return ab, ab_lengths
