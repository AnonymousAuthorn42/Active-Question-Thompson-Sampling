import torch
import numpy as np
import torch.nn as nn


def partial_loss(output, target):
    """
    target: binary encoding of partial labels
    """
    output_prob = nn.functional.softmax(output, dim=-1)
    prob = torch.clamp(torch.sum(output_prob * target, -1),1e-12,1.)
    log_prob = torch.log(prob) # / target.sum(-1)
    return torch.mean(-log_prob)

def CrossEntropyLoss(output, target) :
    output_prob = nn.functional.softmax(output, dim=-1)
    log_ent = torch.sum( torch.log(output_prob) * target, -1)
    return torch.mean(-log_ent)

class HierarchicalLLLoss(nn.Module):
    """
    Hierachical log likelihood loss.
    The weights must be implemented as a nltk.tree object and each node must
    be a float which corresponds to the weight associated with the edge going
    from that node to its parent. The value at the origin is not used and the
    shapre of the weight tree must be the same as the associated hierarchy.
    The input is a flat probability vector in which each entry corresponds to
    a leaf node of the tree. We use alphabetical ordering on the leaf nodes
    labels, which corresponds to the 'normal' imagenet ordering.
    Args:
        hierarchy: The hierarchy used to define the loss.
        classes: A list of classes defining the order of the leaf nodes.
        weights: The weights as a tree of similar shape as hierarchy.
    """

    def __init__(self, CBQ, **kwargs):
        super(HierarchicalLLLoss, self).__init__()
        try :
            self.alpha = kwargs['alpha']
        except :
            self.alpha = 1.
        try : 
            device = kwargs['device']
        except :
            device = 'cpu'
        assert(device !='mps') # mps accelerator not available for now
        Q = torch.tensor(CBQ.questions.Q).float()
        if not CBQ.questions.root :
            Q = torch.vstack((torch.ones((1,10)),Q))
        self.Q = torch.nn.Parameter( Q, requires_grad = False).to(device)
        
        node_dicts = CBQ.questions.node_dicts
        lvls = { 0 : ['root'],
                **{k+1 : list(node_dicts[k].keys()) for k in range(len(node_dicts))}
                }
        class_to_ix = CBQ.questions.treeclass_to_ix
        lvls = {key : [class_to_ix[name] for name in item] for key,item in lvls.items()}
        depth = len(lvls.keys())
        self.weights = torch.nn.Parameter( torch.tensor([np.exp(-self.alpha*k) for k in range(depth)]).float(), 
                                          requires_grad = False ).to(device)
        lvls[max(lvls.keys())+1] = list(CBQ.questions.treeclass_to_actualclass.keys())
        
        self.T_lvl = torch.nn.Parameter(torch.zeros((len(lvls.keys()),CBQ.num_arms +1-CBQ.questions.root )), 
                                                    requires_grad = False ).to(device)
        for key, item in lvls.items():
            for j in item :
                self.T_lvl[key,j] = 1.


    def forward(self, prob, target) :
        idx = torch.stack(torch.where(target>0))
        num = torch.matmul(prob[idx.T[:,0]], self.Q.T)
        new_target =torch.zeros((num.shape[0], prob.shape[-1])).to(prob.device.type)
        #new_target2 =torch.zeros((num.shape[0], prob.shape[-1]))
        #for i,j in zip(torch.arange(idx.shape[-1]), idx.T[:,1]):
        #    new_target[i,j] = 1.
        new_target[torch.arange(idx.shape[-1]),idx.T[:,1]] = 1.
        #assert(new_target ==new_target2).all()
        h = torch.matmul(new_target, self.Q.T)
        #idx2 = h != 0
        #h[idx2] = h[idx2]/h[idx2]
        h = h.clamp(min=0.,max=1.)
        node_probs = torch.multiply(self.T_lvl,h.unsqueeze(1))
        node_probs = torch.bmm(node_probs, num.unsqueeze(-1))
        logs = torch.log(node_probs)
        losses = -torch.matmul(self.weights,torch.diff(logs, dim = 1).transpose(0,2))

        u = torch.zeros(target.shape[0],losses.shape[-1] ).to(prob.device.type)
        #u2 = torch.zeros(target.shape[0],losses.shape[-1] )
        #for i,j in zip(idx[0],torch.arange(losses.shape[-1])):
        #    u[i,j] =1.
        u[idx[0],torch.arange(losses.shape[-1])]=1.
        #assert(u2==u).all()
        losses = torch.matmul(u,losses.T)
        return torch.mean(losses)
        
    def new_weights(self, weights) :
        assert len(self.weights) == len(weights)
        for k in range(len(self.weights)):
            self.weights[k] = weights[k]


class HierarchicalCrossEntropyLoss(HierarchicalLLLoss):
    """
    Combines softmax with HierachicalNLLLoss. Note that the softmax is flat.
    """

    def __init__(self, CBQ, **kwargs):
        super(HierarchicalCrossEntropyLoss, self).__init__(CBQ, **kwargs)


    def forward(self, inputs, index):
        return super(HierarchicalCrossEntropyLoss, self).forward(torch.nn.functional.softmax(inputs, 1), index)


class Partial_HierLoss(nn.Module) :
    def __init__(self, CBQ , **kwargs) :
        super(Partial_HierLoss, self).__init__()
        try :
            self.alpha = kwargs['alpha']
        except :
            self.alpha = 1.
        try : 
            device = kwargs['device']
        except :
            device = 'cpu'
        Q = torch.tensor(CBQ.questions.Q).float()
        if not CBQ.questions.root :
            Q = torch.vstack((torch.ones((1,10)),Q))
        self.Q = torch.nn.Parameter( Q, requires_grad = False).to(device)
        
        node_dicts = CBQ.questions.node_dicts
        lvls = { 0 : ['root'],
                **{k+1 : list(node_dicts[k].keys()) for k in range(len(node_dicts))}
                }
        class_to_ix = CBQ.questions.treeclass_to_ix
        lvls = {key : [class_to_ix[name] for name in item] for key,item in lvls.items()}
        depth = len(lvls.keys())
        self.weights = torch.nn.Parameter( torch.tensor([np.exp(-self.alpha*k) for k in range(depth)]).float(), 
                                          requires_grad = False ).to(device)
        lvls[max(lvls.keys())+1] = list(CBQ.questions.treeclass_to_actualclass.keys())
        
        self.T_lvl = torch.nn.Parameter(torch.zeros((len(lvls.keys()),CBQ.num_arms +1-CBQ.questions.root )), 
                                                    requires_grad = False ).to(device)
        for key, item in lvls.items():
            for j in item :
                self.T_lvl[key,j] = 1.


    def forward(self, logit, target) :
        prob = nn.functional.softmax(logit,-1)
        h = torch.matmul(target, self.Q.T)
        #idx = h != 0
        #h[idx] = h[idx]/h[idx]
        h = h.clamp(min=0.,max=1.)
        num = torch.matmul(prob, self.Q.T)
        node_probs = torch.multiply(self.T_lvl,h.unsqueeze(1))
        node_probs = torch.bmm(node_probs, num.unsqueeze(-1))
        logs = torch.log(node_probs)
        losses = -torch.matmul(self.weights,torch.diff(logs, dim = 1).transpose(0,2))
        return  torch.mean(losses)
    
    def new_weights(self, weights) :
        assert len(self.weights) == len(weights)
        for k in range(len(self.weights)):
            self.weights[k] = weights[k]

    
class Anti_Partial_Loss(nn.Module):
    def __init__(self) :
        super(Anti_Partial_Loss,self).__init__()
    
    def forward(self, output, target):
        output_prob = nn.functional.softmax(output, dim=-1)
        target = torch.ones(target.shape)-target
        prob = torch.sum(output_prob * target, -1) #.clamp(min=1e-8,max=1-1e-8)
        log_prob = torch.log(1.+1e-5-prob)
        return torch.mean(-log_prob)
    

class Composite_loss(nn.Module) :
    def __init__(self, CBQ , beta = 2., **kwargs):
        super(Composite_loss, self).__init__()
        self.loss1 = Partial_HierLoss(CBQ, **kwargs)
        self.beta = beta
        self.loss2 = Anti_Partial_Loss()
        
    def forward(self, logit, target) :
        return self.loss1.forward(logit, target) + self.beta *self.loss2(logit, target)
    
    def new_weights(self,weights):
        self.loss1.new_weights(weights)
    
class Focal_PHXE(nn.Module) :
    def __init__(self, CBQ , tau = 1., **kwargs) :
        super(Focal_PHXE, self).__init__()
        try :
            self.alpha = kwargs['alpha']
        except :
            self.alpha = 1.
        try : 
            device = kwargs['device']
        except :
            device = 'cpu'
        self.tau = tau
        Q = torch.tensor(CBQ.questions.Q).float()
        if not CBQ.questions.root :
            Q = torch.vstack((torch.ones((1,10)),Q))
        self.Q = torch.nn.Parameter( Q, requires_grad = False).to(device)
        
        node_dicts = CBQ.questions.node_dicts
        lvls = { 0 : ['root'],
                **{k+1 : list(node_dicts[k].keys()) for k in range(len(node_dicts))}
                }
        class_to_ix = CBQ.questions.treeclass_to_ix
        lvls = {key : [class_to_ix[name] for name in item] for key,item in lvls.items()}
        depth = len(lvls.keys())
        self.weights = torch.nn.Parameter( torch.tensor([ 1. for k in range(depth)]).float(),  # can be replaced by np.exp(-self.alpha*k)
                                          requires_grad = False ).to(device)
        lvls[max(lvls.keys())+1] = list(CBQ.questions.treeclass_to_actualclass.keys())
        
        self.T_lvl = torch.nn.Parameter(torch.zeros((len(lvls.keys()),CBQ.num_arms +1-CBQ.questions.root )), 
                                                    requires_grad = False ).to(device)
        for key, item in lvls.items():
            for j in item :
                self.T_lvl[key,j] = 1.


    def forward(self, logit, target) :
        prob = nn.functional.softmax(logit,-1)
        h = torch.matmul(target, self.Q.T)
        #idx = h != 0
        #h[idx] = h[idx]/h[idx]
        h = h.clamp(min=0.,max=1.)
        num = torch.matmul(prob, self.Q.T)
        node_probs = torch.multiply(self.T_lvl,h.unsqueeze(1))
        node_probs = torch.bmm(node_probs, num.unsqueeze(-1))
        logs = torch.log(node_probs.clamp(min=1e-12))    #warning if 0. values
        focal_diff_logs = torch.multiply((1+1e-6-node_probs[:,1:,:])**self.tau, torch.diff(logs, dim = 1))
        losses = -torch.matmul(self.weights,focal_diff_logs.transpose(0,2))
        return  torch.mean(losses)
    
    def new_weights(self, weights) :
        assert len(self.weights) == len(weights)
        for k in range(len(self.weights)):
            self.weights[k] = weights[k]

class Focal_PHXE_v2(nn.Module) :
    def __init__(self, CBQ , tau = 1., **kwargs) :
        super(Focal_PHXE_v2, self).__init__()
        try :
            self.alpha = kwargs['alpha']
        except :
            self.alpha = 1.
        try : 
            device = kwargs['device']
        except :
            device = 'cpu'
        self.tau = tau
        Q = torch.tensor(CBQ.questions.Q).float()
        if not CBQ.questions.root :
            Q = torch.vstack((torch.ones((1,10)),Q))
        self.Q = torch.nn.Parameter( Q, requires_grad = False).to(device)
        
        node_dicts = CBQ.questions.node_dicts
        lvls = { 0 : ['root'],
                **{k+1 : list(node_dicts[k].keys()) for k in range(len(node_dicts))}
                }
        class_to_ix = CBQ.questions.treeclass_to_ix
        lvls = {key : [class_to_ix[name] for name in item] for key,item in lvls.items()}
        depth = len(lvls.keys())
        self.weights = torch.nn.Parameter( torch.tensor([ 1. for k in range(depth)]).float(),  # can be replaced by np.exp(-self.alpha*k)
                                          requires_grad = False ).to(device)
        lvls[max(lvls.keys())+1] = list(CBQ.questions.treeclass_to_actualclass.keys())
        
        self.T_lvl = torch.nn.Parameter(torch.zeros((len(lvls.keys()),CBQ.num_arms +1-CBQ.questions.root )), 
                                                    requires_grad = False ).to(device)
        for key, item in lvls.items():
            for j in item :
                self.T_lvl[key,j] = 1.


    def forward(self, logit, target) :
        prob = nn.functional.softmax(logit,-1)
        h = torch.matmul(target, self.Q.T)
        #idx = h != 0
        #h[idx] = h[idx]/h[idx]
        h = h.clamp(min=0.,max=1.)
        num = torch.matmul(prob, self.Q.T)
        node_probs = torch.multiply(self.T_lvl,h.unsqueeze(1))
        node_probs = torch.bmm(node_probs, num.unsqueeze(-1))
        logs = torch.log(node_probs.clamp(min=1e-12))    #warning if 0. values
        focal_diff_logs = torch.multiply( (1e-6-torch.diff(node_probs, dim = 1))**self.tau, torch.diff(logs, dim = 1))
        losses = -torch.matmul(self.weights,focal_diff_logs.transpose(0,2))
        return  torch.mean(losses)
    
    def new_weights(self, weights) :
        assert len(self.weights) == len(weights)
        for k in range(len(self.weights)):
            self.weights[k] = weights[k]