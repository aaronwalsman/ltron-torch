import torch
from torch.distributions import Categorical

def equivalent_outcome_categorical(logits, equivalence):
    b, *c = logits.shape
    device = logits.device
    logits = logits.view(b,-1)
    equivalence = equivalence.view(b,-1)
    
    max_logits, _ = torch.max(logits, dim=1)
    logits = logits - max_logits.view(b,1)
    unnormed_prob = torch.exp(logits)
    
    eq_classes = torch.max(equivalence)+1
    if eq_classes == 1:
        # this turns off gradients
        eq_logits = torch.zeros((b,1))
    else:
        eq_unnormed_prob = torch.full((b, eq_classes), 1e-9, device=device)
        eq_unnormed_prob.scatter_add_(1, equivalence.view(b,-1), unnormed_prob)
        eq_logits = torch.log(eq_unnormed_prob) + max_logits.view(b,1)
    
    return Categorical(logits=eq_logits)
