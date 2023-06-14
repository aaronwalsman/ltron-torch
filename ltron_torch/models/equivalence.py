import torch
import torch.nn.functional as F
from torch.distributions import Categorical

def equivalent_probs_categorical(unnormed_probs, equivalence):
    b, *c = unnormed_probs.shape
    unnormed_probs = unnormed_probs.view(b,-1)
    device = unnormed_probs.device
    eq_classes = torch.max(equivalence)+1
    if eq_classes == 1:
        eq_unnormed_probs = torch.zeros((b,1), device=device)
    else:
        eq_unnormed_probs = torch.zeros((b, eq_classes), device=device)
        eq_unnormed_probs.scatter_add_(
            1, equivalence.view(b,-1), unnormed_probs)
    
    return Categorical(probs=eq_unnormed_probs)

def equivalent_outcome_categorical(
    logits, equivalence, temperature=1., dropout=0., blend_to_avg=None,
):
    b, *c = logits.shape
    device = logits.device
    logits = logits.view(b,-1) * (1./temperature)
    equivalence = equivalence.view(b,-1)
    
    max_logits, _ = torch.max(logits, dim=1)
    logits = logits - max_logits.view(b,1)
    unnormed_prob = torch.exp(logits)
    if dropout:
        unnormed_prob = F.dropout(unnormed_prob, dropout)
    
    eq_classes = torch.max(equivalence)+1
    if eq_classes == 1:
        # this turns off gradients
        eq_logits = torch.zeros((b,1), device=device)
    else:
        eq_unnormed_prob = torch.full((b, eq_classes), 1e-9, device=device)
        eq_unnormed_prob.scatter_add_(1, equivalence.view(b,-1), unnormed_prob)
        
        if blend_to_avg is not None:
            one_everywhere = torch.ones_like(unnormed_prob)
            eq_total = torch.full((b, eq_classes), 1e-9, device=device)
            eq_total.scatter_add_(1, equivalence.view(b,-1), one_everywhere)
            eq_total = eq_total ** blend_to_avg
            eq_unnormed_prob = eq_unnormed_prob / eq_total
        
        eq_logits = torch.log(eq_unnormed_prob) + max_logits.view(b,1)
    
    return Categorical(logits=eq_logits)

def min_max(logits, equivalence, index):
    
    from torch_scatter import scatter_min, scatter_max
    
    b, *c = logits.shape
    logits = logits.view(b,-1)
    equivalence = equivalence.view(b,-1)
    
    #eq_max_logits = torch.zeros((b,eq_classes), device=device)
    #eq_min_loigts = torch.zeros((b,eq_classes), device=device)
    eq_logits, _ = scatter_max(logits, equivalence)
    eq_min_logits, _ = scatter_min(logits, equivalence)
    
    eq_logits[range(b), index] = eq_min_logits[range(b), index]
    
    return Categorical(logits=eq_logits)

def equivalent_mean(logits, equivalence, index):
    from torch_scatter import scatter_mean
    
    b, *c = logits.shape
    logits = logits.view(b,-1)
    equivalence = equivalence.view(b,-1)
    
    #unnormed_probs = torch.exp(logits)
    mean_logits = scatter_mean(logits, equivalence)
    return Categorical(logits=mean_logits)

def avg_equivalent_logits(
    logits, equivalence, temperature=1., dropout=0.,
):
    b, *c = logits.shape
    device = logits.device
    logits = logits.view(b,-1) * (1./temperature)
    equivalence = equivalence.view(b,-1)
    
    total = torch.ones(logits.shape)
    
    eq_classes = torch.max(equivalence)+1
    eq_logits = torch.zeros((b,eq_classes), device=device)
    eq_logits.scatter_add_(1, equivalence.view(b,-1), logits)
    
    eq_total = torch.full((b,eq_classes), 1e-9, device=device)
    eq_total.scatter_add_(1, equivalence.view(b,-1), total)
    
    eq_logits = eq_logits / eq_total
    #eq_logits = torch.nan_to_num(eq_logits, nan=-1e9)
    
    batch_indices = torch.arange(b).view(b,1).expand(equivalence.shape)
    avg_logits = eq_logits[batch_indices, equivalence]
    
    return Categorical(logits=avg_logits)

def avg_equivalent_logprob(logits, equivalence, *indices):
    b, *c = logits.shape
    logprobs = torch.log_softmax(logits.view(b,-1), dim=1).view(logits.shape)
    eq_class = equivalence[(range(b), *indices)]
    eq = equivalence == eq_class.view(b, *[1 for _ in indices])
    sum_eq_logprob = torch.sum((logprobs * eq).view(b,-1), dim=1)
    total = torch.sum(eq.view(b,-1), dim=1)
    avg_logprob = sum_eq_logprob / total
    
    return avg_logprob

def equivalent_inter_entropy(logits, equivalence):
    b, *c = logits.shape
    device = logits.device
    equivalence = logits.view(b,-1)
    
    log_prob = torch.log_softmax(logits, dim=1)
    prob = torch.exp(log_prob)
    lpp = log_prob * prob
    
    #max_logits, _ = torch.max(logits, dim=1)
    #logits = logits - max_logits.view(b,1)
    #unnormed_prob = torch.exp(logits)
    
    eq_classes = torch.max(equivalence)+1
    eq_lpp = torch.full((b, eq_classes), 1e-9, device=device)
    eq_lpp.scatter_add_(1, equivalence.view(b,-1), lpp)
    
    # divisor
    total = torch.ones_like(logits)
    eq_total = torch.zeros((b, eq_classes), device=device)
    eq_total.scatter_add_(1, equivalence.view(b,-1), total)
    
    inter_entropy = -eq_lpp / eq_total
    inter_entropy.view(-1)[torch.isfinite(inter_entropy).view(-1)] = 0
    inter_entropy = inter_entropy.view(b, -1)
    
    return inter_entropy
