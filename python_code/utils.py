import torch
import torch.nn as nn


def hierarchical_accuracy(output, target, lvls, Q) :
    with torch.no_grad():
        probs = output @ Q[1:].T
        pos_labels = target @ Q[1:].T
        pos_labels[pos_labels >0] =1.
        num_classes = target.shape[-1]
        hier_pos = pos_labels.unsqueeze(1) * lvls[1:,1:]
        hier_probs = probs.unsqueeze(1) * lvls[1:,1:]
        positivs = hier_pos.argmax(-1) == hier_probs.argmax(-1)
        corrects = positivs.sum(0) /target.shape[0]
    return corrects      

def get_prob_from_loader(loader, net, device):
    to_stack = []
    net.eval()
    with torch.no_grad():
        for i, data in enumerate(loader):
            X = data[0]
            logit, _ = net(X.to(device))
            to_stack.append( nn.functional.softmax(logit,-1) )
    return torch.vstack(to_stack)


