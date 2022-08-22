import torch
from torch.nn import Embedding
from torch.optim import SGD
from torch.nn.functional import cross_entropy

embedding = Embedding(1, 3)
opt = SGD(embedding.parameters(), lr=0.0001)

def my_cross_entropy(logits, target):
    p = torch.softmax(logits, dim=-1)
    logp = torch.log(p)
    return -torch.sum(logp * target)

print('='*80)
print('Cross Entropy:')
for i in range(3):
    print('-'*80)
    print('Target=%i, Positive:'%i)
    x = torch.LongTensor([0])
    y_hat = embedding(x)
    y = torch.zeros(3)
    y[i] = 1
    ce_loss = my_cross_entropy(y_hat, y)
    ce_loss.backward()
    print(ce_loss)
    print(embedding.weight.grad)
    opt.zero_grad()
    
    print('-'*80)
    print('Target=%i, Negative Uniform:'%i)
    x = torch.LongTensor([0])
    y_hat = embedding(x)
    y = torch.ones(3)/2.
    y[i] = 0
    ce_loss = my_cross_entropy(y_hat, y)
    ce_loss.backward()
    print(ce_loss)
    print(embedding.weight.grad)
    opt.zero_grad()
    
    print('-'*80)
    print('Target=%i, Negative Renormalized:'%i)
    x = torch.LongTensor([0])
    y_hat = embedding(x)
    y = torch.softmax(y_hat, dim=-1).detach()[0]
    y[i] = 0
    y = y / torch.sum(y)
    ce_loss = my_cross_entropy(y_hat, y)
    ce_loss.backward()
    print(ce_loss)
    print(y)
    print(embedding.weight.grad)
    opt.zero_grad()

print('='*80)
print('Policy Gradient:')
for i in range(3):
    for a in [1., -1.]:
        print('-'*80)
        print('Action=%i, Advantage=%f:'%(i,a))
        x = torch.LongTensor([0])
        y_hat = embedding(x)
        p = torch.softmax(y_hat, dim=-1)
        pg_loss = -(torch.log(p[0,i]) * a)
        pg_loss.backward()

        print(pg_loss)
        print(embedding.weight.grad)
        opt.zero_grad()

'''
Conclusion:
There actually IS NOT a target distribution associated with a negative-advantage
example of policy-gradient.  Or rather there is as long as you allow for a
negative scaling factor, in which case the distribution is still just the
one-hot spike, and the scaling factor is the negative-advantage, but this isn't
interesting.  What IS interesting is that zeroing the probability of the bad
action and renormalizing is actually fundamentally different and produces
fundamentally different gradients... which is... weird.  In the short term this
kind of screws up my "what I'm doing is at least as principled as policy
gradients" argument, but maybe opens up even more interesting lines.  What if
zeroing, renormalizing and doing cross-entropy is just better?  Should we expect
it to be?  I want to say that it's nice because we're supervising it with a
distribution, but at the same time, the softmax always ensures that the result
is a distribution.  It's also nice because it lets us modify the target as we
see fit (zeroing out known-to-be-bad actions)... but now it's less principled.
But it also might just work better.  And might also be easier to tune?  Who
knows, this is all wild stuff.  And maybe this approach is already known, who
knows?

Oh ok, you know what though?  We can still get something out of this.  Here's
what we do: the policy gradient update still produces a set of distributions
depending on the advantage and the learning rate.  We can plot those
distributions as a function of advantage + learning rate.  What do they look
like?  Are they better or worse than our target distributions?  Or rather are
they better or worse than the result of taking a gradient step using our target
distribution?

This kind of looks like Boltzmann exploration too, right?  Except that instead
of taking the mean of the rewards, I'm coming up with this target distribution.
Is this the same thing though?  Maybe?  I have to think about it I guess.
Check out:
https://proceedings.neurips.cc/paper/2017/file/b299ad862b6f12cb57679f0538eca514-Paper.pdf
'''
