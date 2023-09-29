import numpy as np
import torch
from hierarchy import *
from torch.distributions import Categorical

def entropy_calc(y_hat):
    eps = 1e-7 #originally : 4
    y_hat = torch.clamp(y_hat, min=eps, max=1-eps)
    if y_hat.shape[1] == 1:
        return -((y_hat * torch.log(y_hat)) +
                 ((1-y_hat) * torch.log(1-y_hat)))
    else:
        return Categorical(probs=y_hat).entropy()
    
def marginal_entropy_calc(pr, y=None):
    eps = 1e-7
    pr = pr.clamp(min = eps, max = 1-eps)
    if y is None :
        y = torch.ones_like(pr)
        norm = torch.ones(pr.shape[-1])
    else :
        norm = (pr * y).sum(-1)
    if y.dim()>1 : plogp = pr/norm.unsqueeze(1) * torch.log(pr/norm.unsqueeze(1))
    else : plogp = pr/norm * torch.log(pr/norm)
    plogp = torch.nan_to_num(plogp)
    return - torch.nan_to_num((plogp * y)).sum(-1)

class Questions:
    def __init__(self, args, label_to_idx, **kwargs):
        self.args = args
        self.dataset = self.args.dataset_name
        if self.args.dag_hier :
            self.node_dicts = {'TOY': TOY_node_dicts,
                        'TOY2': TOY_node_dicts,
                        'FashionMNIST' : FASHIONMNIST_node_dicts_2,
                        'CIFAR10' : CIFAR10_node_dicts_2,
                        'CIFAR100': CIFAR100_node_dicts_2,
                        'tiny-imagenet-200' : TIN_node_dict_2}[self.dataset]
            print('second hierarchy selected')
        else :
            self.node_dicts = {'TOY': TOY_node_dicts,
                        'TOY2': TOY_node_dicts,
                        'FashionMNIST' : FASHIONMNIST_node_dicts,
                        'CIFAR10' : CIFAR10_node_dicts,
                        'CIFAR100': CIFAR100_node_dicts,
                        'tiny-imagenet-200' : TIN_node_dict}[self.dataset]
            print('first hierarchy selected')
        self.root = self.args.inactive_arm
        self.num_classes=len(label_to_idx)
        if 'random_tree' in kwargs.keys():
            if kwargs['random_tree'] == True :
                try :
                    depth = kwargs['tree_depth']
                except :
                    depth = 3 
                self.node_dicts = random_hierarchy(self.num_classes, depth, class_names=list(label_to_idx.keys()) )
                print('random depth', len(self.node_dicts)+1)
        if 'node_dicts' in kwargs.keys():
            self.node_dicts = kwargs['node_dicts']
        
        
        self.context_type = args.contexts
        self.treeclass_to_ix, self.actualclass_to_treeclass, self.tree_to_actual = compute_hierarchy(self.node_dicts, label_to_idx)
        self.ix_to_treeclass = {self.treeclass_to_ix[k]: k for k in self.treeclass_to_ix}
        self.treeclass_to_actualclass = {self.actualclass_to_treeclass[k]: k for k in self.actualclass_to_treeclass}
        self.num_questions = len(self.treeclass_to_ix.keys()) - (0 if self.root else 1)
        self.Q = np.zeros((self.num_questions, self.num_classes))
        self.leaves = list(self.actualclass_to_treeclass.values())
        
        
        lvls = { 0 : ['root'],
                **{k+1 : list(self.node_dicts[k].keys()) for k in range(len(self.node_dicts))}
                }
        lvls = {key : [self.treeclass_to_ix[name] for name in item] for key,item in lvls.items()}
        depth = len(lvls.keys())
        lvls[max(lvls.keys())+1] = list(self.treeclass_to_actualclass.keys())
        
        self.T_lvl = np.zeros((len(lvls.keys()),self.num_questions +1-self.root ))
                                                    
        for key, item in lvls.items():
            for j in item :
                self.T_lvl[key,j] = 1.
        
        self.cost_annotation(**kwargs)
        print(self.costs)


    def generate_binary_questions(self):
        for i in self.tree_to_actual.keys():
            if self.root :
                self.Q[i, self.tree_to_actual[i]] = 1 #root not removed
            else :
                if i > 0 :
                   self.Q[i-1, self.tree_to_actual[i]] = 1   #i-1 because root is removed


    
    def context_generation(self, pr, y_partial, dataset_labels, pr_val, y_val, emb_val, emb):
        dim = y_partial.dim()
        dataset_labels = dataset_labels[dataset_labels.sum(-1)<self.num_classes]
        if self.context_type is None :
            self.context_type = ['eig','cost','align']
        
        
        dimensions = {'eig' : self.eig(pr), #expected information gain
                      'cost' : self.costs if dim ==1 else self.costs.repeat(pr.shape[0]).reshape(-1,pr.shape[0]).T ,
                      'ent' : entropy_calc(pr.unsqueeze(0)).numpy().repeat(self.num_questions).reshape(-1,self.num_questions) ,
                      'erc' : self.erc(pr,y_partial) , # expected remaining classes
                      'edc' :  (y_partial.numpy().sum(-1) - self.erc(pr,y_partial).T).T, # expected decrease in classes
                      'bias' : np.ones(self.num_questions) if dim ==1 else np.ones((pr.shape[0],self.num_questions)),
                      'lc' : (1 - pr.numpy().max(-1)).repeat(self.num_questions) if dim ==1 else (1 - pr.numpy().max(1)).repeat(self.num_questions).reshape(-1,self.num_questions), # uncertainty of the current prediction
                      'lcq' : 1 -np.abs( np.dot(pr,self.Q.T) - np.dot(pr,1-self.Q.T) ),  #uncertainty of the answer (yes or no) for each question
                      'n_lab' : y_partial.sum(-1).repeat(self.num_questions) if dim ==1 else y_partial.sum(-1).repeat(self.num_questions).reshape(-1,self.num_questions), #current size of partial class
                      'ent_m' : marginal_entropy_calc(pr, y_partial).numpy().repeat(self.num_questions),
                      'eig_m' : self.marginal_eig(pr,y_partial) if dim>1 else self.marginal_eig(pr.unsqueeze(0), y_partial.unsqueeze(0)).reshape(-1)
                     }
        
        # conditions to avoid the calculation of unnecessary computationally expensive variables
        if 'delta_loss' in self.context_type :
            dimensions['delta_loss'] = self.delta_loss_estimate(pr_val, y_val, emb_val, pr, y_partial, emb)
        if 'prop_q' in self.context_type : 
            dimensions['prop_q']= (dataset_labels @ self.Q.T).clip(min=0.,max=1.).mean(0) if dim==1 else (dataset_labels @ self.Q.T).clip(min=0.,max=1.).mean(0).repeat(y_partial.shape[0]).reshape(y_partial.shape[0],-1)
        if 'dispq' in self.context_type :
            dimensions['dispq'] = (1 - (dataset_labels @ self.Q.T).clip(min=0.,max=1.).mean(0)) if dim==1 else (1 - (dataset_labels @ self.Q.T).clip(min=0.,max=1.).mean(0)).repeat(y_partial.shape[0]).reshape(y_partial.shape[0],-1),
    
        #computing entropies in the hierarchy
        lvls = self.T_lvl[1:,1:]
        lvls = torch.tensor(lvls).float()
        Q = (self.Q[self.root:]).astype('float32')
        total_probs = pr @ Q.T
        lvls_probs = total_probs *lvls.unsqueeze(1)
        ents = []
        for i, tens in enumerate(lvls_probs) :
            t = tens[:,lvls[i]!=0] 
            if dim == 1 : t = t.squeeze(0)
            ent = entropy_calc(t.unsqueeze(0)).numpy().repeat(self.num_questions).reshape(-1,self.num_questions)
            dimensions[f'ent_{i+1}'] = ent
            if 'hier_ent' in self.context_type and f'ent_{i+1}' not in self.context_type :
                self.context_type.append(f'ent_{i+1}')
        if 'hier_ent' in self.context_type :
            self.context_type.remove('hier_ent')


        if dim == 1 :
            return np.vstack([dimensions[c] for c in self.context_type])
        else :
            return np.stack([dimensions[c] for c in self.context_type]).transpose((1,0,2))
    
    
    def eig(self, pr):
        qp = np.dot(pr, self.Q.T)
        qp = np.clip(qp,a_min=1e-12,a_max=1.-1e-12)
        mere = np.nan_to_num(-(qp*np.log(qp)+(1-qp)*np.log(1-qp))) 
        return mere
    
    def erc(self,pr, y_partial):
        qp = np.dot(pr, self.Q.T)
        y_1 = np.dot(y_partial.numpy(),self.Q.T) 
        y_0 = (y_1.max(-1)- y_1.T).T              
        return qp*y_1 + (1-qp)*y_0
    
    def grad(self, pr, y_partial):
        if pr.dim() == 1 :
            return pr - torch.nan_to_num(pr * y_partial / (pr*y_partial).sum(-1).unsqueeze(1))
        else :
           return  pr- torch.nan_to_num(pr * y_partial / (pr*y_partial).sum(-1).unsqueeze(-1))

       
    def delta_loss_estimate(self, pr_val, y_val, emb_val, pr, y_partial, emb ):
        Q = torch.tensor(self.Q).float()
        if pr.dim() == 1 :
            y_1 = y_partial * Q
            y_0 = y_partial - y_1
            qp = pr @ Q.T
            grad_val = self.grad(pr_val,y_val)
            prods = (emb_val *emb ).sum(-1)+1
            grad_example = (qp.unsqueeze(1)*torch.nan_to_num(self.grad(pr,y_1))+(1-qp).unsqueeze(1)*torch.nan_to_num(self.grad(pr,y_0))).unsqueeze(1)
            su = (prods.unsqueeze(1) * grad_val @ grad_example.transpose(1,2)).sum(1).squeeze(1)
            return 30*0.0001 * su.detach() / len(pr_val)
        elif pr.dim() == 2 :
            y_1 = y_partial.unsqueeze(1) * Q
            y_0 = y_partial.unsqueeze(1) - y_1
            qp = pr @ Q.T
            grad_val = self.grad(pr_val,y_val)
            prods = (emb_val * emb.unsqueeze(1) ).sum(-1)+1
            grad_example = (qp.unsqueeze(-1)*torch.nan_to_num(self.grad(pr.unsqueeze(1),y_1))+ \
                            (1-qp).unsqueeze(-1)*torch.nan_to_num(self.grad(pr.unsqueeze(1),y_0))).unsqueeze(2)
            su = (prods.unsqueeze(-1).unsqueeze(1) * grad_val @ grad_example.transpose(2,3)).sum(2).squeeze(-1)
            return 30*0.0001 * su.detach() / len(pr_val)
        
    
    
    def marginal_eig(self,pr, y_partial):
         y_1 = y_partial.unsqueeze(1) * torch.tensor(self.Q).float()
         qp = (y_1 * pr.unsqueeze(1)).sum(-1) / (y_partial * pr).sum(-1).unsqueeze(1)
         return -np.nan_to_num(qp *torch.log(qp)+ (1-qp)*torch.log(1-qp))
    

        
    def answer_question(self, selected_arm, rnd_samp, training_data) :
        true_y = training_data.data.targets[rnd_samp]
        y_partial =training_data.partial_labels[rnd_samp].cpu()
        
        response = true_y in self.tree_to_actual[selected_arm +(0 if self.root else 1)]
        if response :
            answer = y_partial.numpy() * self.Q[selected_arm]
        else:
            answer =  y_partial.numpy()-y_partial.numpy()*self.Q[selected_arm] 
            
        training_data.update_label( rnd_samp, torch.from_numpy(answer) )
        return answer, response

    def adjacency_matrix(self):
        num_nodes = max([ item for _,item in self.treeclass_to_ix.items()])+1
        M = np.zeros((num_nodes,num_nodes))
        as_parent = []
        for dictionnary in self.node_dicts :
            for key, childlist in dictionnary.items():
                i = self.treeclass_to_ix[key]
                for child in childlist :
                    j = self.treeclass_to_ix[child]
                    M[ i, j ] = 1
                    as_parent.append( j )
        for k in range(1,num_nodes):
            if k not in as_parent :
                M[0,k] = 1
        return M    

    def cost_annotation(self) :
        M = self.adjacency_matrix()
        depth= np.zeros(len(M))
        temp = np.eye(len(M))
        k = 0
        while (temp[0] != np.zeros(len(M))).any():
            temp = temp @ M
            k+=1
            for node in np.where(temp[0] != 0)[0]:
                if depth[node] == 0 : depth[node] = k
        
        def linear_cost(array) :
            return self.args.leaf_cost * array / array.max()
        
        def my_cost_func(array):
            '''
            define your own cost function here
            '''
            max_depth = array.max()
            return (1/ (self.args.leaf_cost**(max_depth-1)) ) * self.args.leaf_cost**array  

        self.costs = (1 - self.args.alpha) * linear_cost(depth) + self.args.alpha * my_cost_func(depth)
        self.costs[0] = 0. # inactive arm cost is 0
        for _, item in self.actualclass_to_treeclass.items() : #leaf costs are all equal
            self.costs[item] = self.args.leaf_cost
