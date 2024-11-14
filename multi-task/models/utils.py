import torch
import numpy as np
import copy
from torch.autograd import Variable

def to_var(x, requires_grad = False, volatile = False):
    if torch.cuda.is_available():
        x = x.to(torch.device("cuda"))
    return Variable(x, requires_grad = requires_grad, volatile = volatile)

def count_weights(net):
    num = 0
    for p in net.parameters():
        if len(p.data.size()) != 1:
            num = num + p.numel()
    return num

def count(weight_masks):
    n = 0
    for i in range(len(weight_masks)):
        n = n + torch.sum(weight_masks[i])
    return n

def rand_prune_masks(network, prune_perc, dev):
    net = copy.deepcopy(network)
    net.to(dev)
    net.train()

    scores = []
    for p in net.parameters():
        if len(p.data.size()) != 1:
            print(p.data.size())
            scores.append(torch.rand(p.data.size()))

    all_weights = []
    for i in range(len(scores)):
        all_weights += list(scores[i].cpu().data.abs().numpy().flatten())
    threshold = np.percentile(np.array(all_weights), prune_perc)

    weight_masks = []
    for i in range(len(scores)):
        pruned_inds = scores[i] > threshold
        weight_masks.append(pruned_inds.float())

    bias_masks = []
    for i in range(len(weight_masks)):
        mask = torch.ones(len(weight_masks[i]))
        for j in range(len(weight_masks[i])):
            if torch.sum(weight_masks[i][j]) == 0:
                mask[j] = torch.tensor(0.0)
        mask.to(dev)
        bias_masks.append(mask)
    print(1.0 - (count(weight_masks)) / (count_weights(net)))
    del net
    return weight_masks, bias_masks

def gumbel_sigmoid(logits, k: int = 2, hard: bool = False, dim: int = 1, use_gumbel=True, tau: float = 1.0) : 
    gumbels = ( -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()) 
    if use_gumbel:
        gumbels = (logits + gumbels) / tau  
    else:
        if tau == None:
            gumbels = logits
        else:
            gumbels = logits / tau
    y_soft = gumbels.sigmoid()

    if hard: 
        indices = torch.topk(y_soft, k, dim=dim)[-1]
        y_hard1 = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard2 = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard1.scatter_(1, indices[:,0].view(indices[:,0].size(0),1, indices[:,0].size(1)), 1)
        y_hard2.scatter_(1, indices[:,1].view(indices[:,1].size(0),1, indices[:,1].size(1)), 1)
        ret1 = y_hard1 - y_soft.detach() + y_soft
        ret2 = y_hard2 - y_soft.detach() + y_soft
    else:
        ret1 = y_soft
        ret2 = y_soft
    return ret1, ret2

def gumbel_softmax(logits, hard: bool = False, dim: int = 1, use_gumbel=True, tau: float = 1.0):
    gumbels = ( -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()) 
    if use_gumbel:
        gumbels = (logits + gumbels) / tau  
    else:
        if tau == None:
            gumbels = logits
        else:
            gumbels = logits / tau
    y_soft = gumbels.softmax(dim=dim)

    if hard:
        indices = torch.topk(y_soft, 1, dim=dim)[-1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard.scatter_(1, indices[:,0].view(indices[:,0].size(0),1, indices[:,0].size(1)), 1)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret