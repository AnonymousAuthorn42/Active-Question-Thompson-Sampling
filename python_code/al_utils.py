import os
import pickle
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group
from datasets import MyDataset, preprocess_data, train_val_split
from networks import  Trainer, mlpMod,  weight_reset, small_VGG, VGG, LeNet5, ConvNet
from resnets import def_resnet
from losses import partial_loss
from questions import entropy_calc, marginal_entropy_calc, Questions


class ContextualBanditsQuestions:
    def __init__(self, args, questions : Questions):
        
        self.questions = questions
        self.context_dim = len(self.questions.context_type) + ('pos' in self.questions.context_type)
        if 'hier_ent' in self.questions.context_type :
            self.context_dim += - 2 +self.questions.T_lvl.shape[0]
        self.questions.generate_binary_questions()
        self.arms_data = self.questions.Q


        self.num_arms = self.questions.num_questions
        self.gamma_forget = args.gamma_forget
        self.lambda_regul = args.lambda_regul
        self.bc = np.zeros(self.context_dim)

        # thompson sampling params
        self.nu2 = args.nu2
        self.B = self.lambda_regul * np.eye(self.context_dim)
        self.B_inv = self.B / self.lambda_regul
        self.var = self.nu2 * self.B_inv
        self.mu_cap = np.zeros((self.context_dim, 1))
        self.mu_cap[np.where(np.array(self.questions.context_type)=='cost')[0]] = args.lambda_ann
        self.mu_bar = np.zeros((self.context_dim, 1))
        self.f = np.zeros((self.context_dim, 1))

        self.B_tmp = np.zeros_like(self.B)
        self.f_tmp = np.zeros_like(self.f)

    def sample_mu(self):
        self.mu_bar = np.random.multivariate_normal(self.mu_cap[:, 0], self.var)

    def update_tsp_tmp(self, reward, i):
        if i == 0:
            self.B_tmp = np.zeros_like(self.B)
            self.f_tmp = np.zeros_like(self.f)
            self.B_inv_tmp = self.B_inv
        outer_prod = np.expand_dims(self.bc,1) @ np.expand_dims(self.bc,0) 
        self.B_tmp += outer_prod
        self.f_tmp += np.expand_dims(self.bc, axis=1) * reward 
        self.B_inv_tmp = self.B_inv_tmp - self.B_inv_tmp  @ outer_prod @ self.B_inv_tmp / ( 1 + self.bc.T @ self.B_inv_tmp @ self.bc ) 
        
        
    def update_thompson_sampling(self):
        self.B = (self.gamma_forget*self.B) + self.B_tmp
        self.f = (self.gamma_forget*self.f) + self.f_tmp
        self.B_inv = self.B_inv_tmp
        self.mu_cap = self.B_inv @ self.f
        self.var = self.nu2 * self.B_inv
        
    def rem_q(self, answer):
        return sorted(np.where( (answer @ self.arms_data.T >0) * (answer@ self.arms_data.T <answer.sum()) )[0].tolist()+ [0])

    def choose_arm(self, question_sample_context, sample_ques, rnd_samp, 
                                  mode, all_arms_available) :
        if not all_arms_available :
            if mode == 'bandit':
                selected_idx = np.argsort(question_sample_context[:, sample_ques[rnd_samp]['remaining_ques']].T @ self.mu_bar)[-1]
            elif mode == 'eig':
                selected_idx = np.argsort(question_sample_context[0, sample_ques[rnd_samp]['remaining_ques']])[-1]
            elif mode == 'random':
                selected_idx = np.random.randint( len(sample_ques[rnd_samp]['remaining_ques']) ) #permutation(np.arange(len(sample_ques[rnd_samp]['remaining_ques'])))[0]
            selected_arm = sample_ques[rnd_samp]['remaining_ques'][selected_idx]
        else :
            if mode == 'bandit':
                selected_idx = np.argsort(question_sample_context.T @ self.mu_bar)[-1]
            elif mode == 'eig':
                selected_idx = np.argsort(question_sample_context[0,:])[-1]              
            elif mode == 'random':
                selected_idx = np.random.randint(self.num_arms) #permutation(np.arange(self.num_arms))[0]
            selected_arm = np.arange(self.num_arms)[selected_idx]
            assert selected_arm == selected_idx
        return selected_arm
    


def invert_permutation(p):
    """Return an array s with which np.array_equal(arr[p][s], arr) is True.
    The array_like argument p must be some permutation of 0, 1, ..., len(p)-1.
    """
    p = np.asanyarray(p) # in case p is a tuple, etc.
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s

def argsort_as(input, as_output):
    s = invert_permutation(np.argsort(as_output))
    return np.argsort(input)[s]

def ddp_setup(rank :int, world_size : int):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank = rank, world_size = world_size)
    torch.cuda.set_device(rank)


class AL_process :
    def __init__(self, args):
        self.args = args
        if self.args.device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
            self.device = torch.device(device)
        else :
            self.device = torch.device(self.args.device)

        
        preprocess = preprocess_data(args.architecture, args.resnet_type, args.dataset_name)

        self.train_data = MyDataset(args.dataset_name, args.data_dir, train_bool = True, preprocess = preprocess)
        self.test_data = MyDataset(args.dataset_name, args.data_dir, train_bool = False, preprocess = preprocess)
        self.train_set, self.val_set = train_val_split(self.train_data, args.val_prop)
        self.train_data.set_val_indices(self.val_set)
        self.test_data.true_labels()
        self.eval_val = True

        FMQ = Questions(args, hyp_emb_file = None, label_to_idx = self.train_data.class_to_idx)
        self.CBQ = ContextualBanditsQuestions( args, questions = FMQ )
        self.stats = {'anno_cnt': [0],
                        'partial_cnt' : [],
                        'full_cnt' : [],
                        'inactive_cnt' :[],
                        'mu_array' : self.CBQ.mu_cap,
                        'mu_cov' : [self.CBQ.var],
                        'test_acc' : [],
                        'test_loss' :[],
                        'reward_vects' :[],
                        'label_type_cnt' : {'FL' : [0],
                                            'PL' : [0],
                                            'UL' : [len(self.train_set.indices)]},
                        'contexts' : [],
                        'config' : args
                    }
        
        self.lambdas = np.array([args.lambda_acc,
                                 args.lambda_loss,
                                 args.lambda_aig,
                                 args.lambda_aig_m,
                                 args.lambda_cos,
                                 args.lambda_ann,
                                 args.lambda_loc_loss])
        
        self.init_loss = 0.
        self.init_acc = 0.
        self.new_loss = 0.
        self.new_acc = 0.
        temp_dirs = [l for l in os.listdir() if 'temp' in l and os.path.isdir(l)]
        k = 0
        while f'temp_{k}' in temp_dirs :
            k+=1
        self.temp_dir = f'temp_{k}'
        print('temporary directory :',self.temp_dir)
        os.mkdir(self.temp_dir)
        if self.args.pool_is_X and self.args.allow_multichoice  and not self.args.explore_all : algo_name = 'AQTS-R'
        elif not self.args.pool_is_X and not self.args.allow_multichoice and not self.args.explore_all : algo_name = 'AQTS-CR'
        elif self.args.explore_all and not not self.args.pool_is_X and not self.args.allow_multichoice : algo_name = 'AQTS-E'
        elif self.args.mode == 'random' : algo_name = 'AQ-R'
        elif self.args.mode == 'eig'  : algo_name = 'AQ-EIG'
        else : algo_name = 'AQTS-FA'
        self.file_name = f'{self.args.dataset_name}_{self.args.architecture}_{algo_name}.pkl'

    def clear_temp_path(self) :
        shutil.rmtree(self.temp_dir)


    def initialize_process(self):
        if self.args.initialize :
            if self.args.proportion :
                init_idx = np.random.permutation(self.train_set.indices)
                init_idx = init_idx[:int(self.args.init_lab_prop*len(init_idx))]
                
            else :
                init_idx = np.random.permutation(self.train_set.indices)[:self.args.init_lab_size]
            self.train_data.true_labels(init_idx)
            self.stats['label_type_cnt']['FL'] = [len(init_idx)]
            self.stats['label_type_cnt']['UL'][0] -= len(init_idx)
        self.sample_ques = {i: {'remaining_ques': np.arange(self.CBQ.questions.num_questions)[self.args.reject_inactive_rnd_1:], 
                           'prev_ques':[], 
                           'prev_ans':[], 
                           'remove_ques': []} for i in self.train_set.indices}
        if self.args.initialize :
            for idx in init_idx:
                self.sample_ques[idx]['remaining_ques']=[0]
                self.sample_ques[idx]['remove_ques']= np.arange(1,self.CBQ.num_arms)
        

    def define_loader(self, dataset = None, train = True ) :
        train_idx = self.train_data.train_indices()
        shuffle = torch.cuda.device_count() <= 1
        
        if dataset is None :
            training_set = Subset(self.train_data, train_idx)
        else : 
            training_set = dataset
        if self.device == torch.device('cpu') or self.device == torch.device('mps') :
            current_loader = DataLoader(training_set, batch_size = self.args.batch_size,
                                        shuffle = train, pin_memory = True,
                                        num_workers = self.args.num_threads,
                                        drop_last = train)
        else :
            if train and shuffle :
                current_loader = DataLoader(training_set, batch_size = self.args.batch_size,
                                            shuffle = shuffle, pin_memory = True, 
                                            num_workers = self.args.num_threads )
            elif train :
                current_loader = DataLoader(training_set, batch_size = self.args.batch_size // torch.cuda.device_count(),
                                            shuffle = False, pin_memory = True, 
                                            sampler = DistributedSampler(training_set) )
            elif not train :
                current_loader = DataLoader(training_set, batch_size = 250, drop_last = False,
                                            shuffle = False, pin_memory = True, 
                                            sampler = DistributedSampler(training_set, shuffle = False, num_replicas = torch.cuda.device_count() ))
            
        return current_loader
        
    def define_model(self, net_name = None):
        if net_name is None : net_name = self.args.architecture
        if net_name == 'mlp' : 
            self.net = mlpMod(dim = self.train_data.img_dim, nclasses = self.train_data.num_classes, embSize = 256)
        elif net_name =='resnet' : 
            self.net = def_resnet(self.args.resnet_type, num_classes = self.train_data.num_classes)
        elif 'small_VGG' in net_name : 
            self.net = small_VGG(vgg_name = net_name, num_classes  = self.train_data.num_classes)
        elif 'VGG' in net_name : 
            self.net = VGG(vgg_name = net_name, num_classes  = self.train_data.num_classes)
        elif net_name =='lenet5' : 
            grayscale = self.args.dataset_name =='FashionMNIST'
            self.net = LeNet5(num_classes=self.train_data.num_classes, grayscale=grayscale, embSize=256)
        elif net_name == 'convnet' :
            in_channels = 1 if self.args.dataset_name =='FashionMNIST' else 3
            self.net = ConvNet(in_channels = in_channels, num_classes = self.train_data.num_classes, img_dim = self.train_data.img_dim)
        
        self.net.apply(weight_reset)
        
        self.loss = partial_loss

        if self.args.optimizer == 'adam' :
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.args.lr, 
                                    betas=(self.args.beta1,self.args.beta2))
        elif self.args.optimizer == 'sgd' :
            self.optimizer = optim.SGD(self.net.parameters(), lr = self.args.lr,
                                       momentum=self.args.momentum)
        else :
            raise NameError('Please implement this optimizer yourself')
        

    def run_batch(self, inputs, target) :
        self.optimizer.zero_grad()
        outputs, _ = self.net(inputs)
        loss = self.loss(outputs, target)
        loss.backward()
        self.optimizer.step()
        
    def run_epoch(self, epoch, loader) :
        try :
            loader.sampler.set_epoch(epoch)
        except :
            if False : print(self.device)
        for inputs, targets, idx in loader :
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            self.run_batch(inputs, targets)
            
    def train_model(self, max_epochs):
        loader = self.define_loader()
        self.net.train()
        for epoch in range(max_epochs):
            self.run_epoch(epoch, loader)
        
        test_loader = self.define_loader(self.test_data, train = False)
        self.get_prob_and_emb(test_loader, style = 'test')
        if self.eval_val :
            val_loader = self.define_loader(self.val_set, train = False)
            self.get_prob_and_emb(val_loader, style = 'validation')
        

    def multigpu_train(self, rank, world_size, n_epoch) :
        ddp_setup(rank, world_size)
        test_loader = self.define_loader( dataset = self.test_data, train = False)
        loader = self.define_loader()
        self.net.train()
        trainer = Trainer(self.net, loader, self.optimizer, self.loss, self.temp_dir, rank)
        trainer.train(n_epoch)
        trainer.eval(test_loader, 'test')
        if self.eval_val :
            val_loader = self.define_loader(dataset = self.val_set, train = False)
            trainer.eval(val_loader, 'validation')
        if rank == 0 : 
            torch.save(trainer.model.module.state_dict(), os.path.join(self.temp_dir, 'current_model.pt') ) 
        destroy_process_group()

    def get_prob_and_emb(self, loader, style ) :
        self.net.eval()
        logits, embs, idxs = [], [], [] 
        val_loss = torch.Tensor([0.]).to(self.device)
        val_acc = torch.Tensor([0.]).to(self.device)
        with torch.no_grad():
            for inputs, targets, idx in loader :
                output, emb = self.net( inputs.to(self.device) )
                targets = targets.to(self.device)
                loss = self.loss(output, targets)
                val_loss += loss * len(inputs) / (len(loader.dataset))
                val_acc += (output.argmax(-1) == targets.argmax(-1)).sum() / len(loader.dataset)
                logits.append(output)
                embs.append(emb)
                idxs.append(idx)
        logits = torch.vstack(logits)
        embs = torch.vstack(embs) 
        idxs = torch.hstack(idxs)
        if style in ['validation', 'inputs'] :
            elts = {'prob' : nn.functional.softmax(logits, dim=-1),
                    'emb' : embs,
                    'idx' : idxs,                  
                    'loss' : val_loss,
                    'acc' : val_acc}
        elif style =='test' :
            elts = {'test_acc' : val_acc,
                    'test_loss' : val_loss}
        file_name = os.path.join(self.temp_dir, f'{style}.pkl')
        with open(file_name,'wb') as f:
            pickle.dump(elts, f)

    def get_prob_and_emb_multi(self, rank, world_size, dataset, style) :
        ddp_setup(rank, world_size)
        loader = self.define_loader(dataset, train = False)
        self.net.eval()
        trainer = Trainer(self.net, loader, self.optimizer, self.loss, self.temp_dir, rank)
        trainer.eval(loader, style)
        destroy_process_group()
    
    def train(self, world_size):
        if world_size >= 1 and self.args.device not in ['cpu','mps'] : 
            mp.spawn(self.multigpu_train, args = (world_size, self.args.num_epochs,), nprocs = world_size)
            state = torch.load(os.path.join(self.temp_dir, 'current_model.pt') )
            self.net.load_state_dict(state)
        else : 
            self.net.to(self.device)
            self.train_model(self.args.num_epochs)

    def compute_prob_and_emb(self, world_size, dataset, rnd_shuffle, style = 'inputs'):
        if world_size > 1 and self.args.device != 'cpu' : 
            mp.spawn(self.get_prob_and_emb_multi, 
                        args = (world_size, dataset, style, ), 
                        nprocs = world_size)
            prob_mat, emb_mat, input_idx = self.load(style)
            reorder = argsort_as(input_idx, rnd_shuffle)
            prob_mat = prob_mat[reorder]
            emb_mat = emb_mat[reorder]
        else : 
            self.net.to(self.device)
            input_loader = self.define_loader(dataset = dataset, train = False)
            self.get_prob_and_emb(input_loader, style)
            prob_mat, emb_mat, _ = self.load(style)
        return prob_mat, emb_mat

    
        
    def get_performances(self):
        with open(os.path.join(self.temp_dir,'validation.pkl'), 'rb') as f:
            stat = pickle.load(f)
        return stat['acc'].cpu().item(), stat['loss'].cpu().item()


    def get_batch_example(self, batch_size = None):
        if batch_size is None :
            batch_size = self.args.num_pts
        if not self.args.explore_all :
            if self.args.pool_is_X :
                if self.args.allow_multichoice :
                    return np.random.choice(self.train_set.indices, batch_size)
                else :
                    return np.random.permutation(self.train_set.indices)[:batch_size]
            else :
                remaining_idx = self.train_data.remaining_samples()
                if self.args.allow_multichoice :
                    return np.random.choice(remaining_idx, batch_size)
                else : 
                    return np.random.permutation(remaining_idx)[:batch_size]
        if self.args.explore_all :
            try :
                self.batch_cnt += 1
                return self.batches[self.batch_cnt] 
            except :
                indices = self.train_set.indices if self.args.pool_is_X else self.train_data.remaining_samples()
                permuted_idx = np.random.permutation(indices)
                self.batches = [ permuted_idx[batch_size*i:batch_size*(i+1)] for i in range(len(permuted_idx)//batch_size) ]
                self.batch_cnt = 0
                return self.batches[self.batch_cnt]

    def init_answering_process(self):
         self.current_labels = self.train_data.partial_labels.clone()
         self.all_contexts = []
         self.all_arms = [] 
         self.local_anno_cost = []
         self.stats['partial_cnt'].append(0)
         self.stats['full_cnt'].append(0)
         self.stats['inactive_cnt'].append(0)
         self.stats['anno_cnt'].append(0)
         
         


    def get_question(self, i, idx, prob_mat, emb_mat, pr_val, emb_val) :
        self.CBQ.sample_mu()
                
        y_partial = self.train_data.partial_labels[idx].cpu()
        partials = self.train_data.partial_labels[self.train_set.indices].cpu()
        
        Y_val = self.train_data.partial_labels[self.val_set.indices].cpu()
        
        if self.args.mode != 'random' :
            question_sample_context = self.CBQ.questions.context_generation(prob_mat[i], y_partial, partials,                                                                            
                                                                    pr_val, Y_val, emb_val, emb_mat[i] )
        else :
            question_sample_context = np.zeros((self.CBQ.context_dim,self.CBQ.num_arms))
        if self.args.inactive_zero_context :
            question_sample_context[:,0] = 0
        # SELECT QUESTION - BANDIT PART HERE
        selected_arm = self.CBQ.choose_arm(question_sample_context, self.sample_ques, idx, 
                                            self.args.mode, self.args.all_arms_available)
        
        self.all_contexts.append(question_sample_context[:,selected_arm])
        self.all_arms.append(selected_arm)
        self.local_anno_cost.append(self.CBQ.questions.costs[selected_arm])
        self.stats['anno_cnt'][-1] += self.CBQ.questions.costs[selected_arm]
        if selected_arm == 0 and self.args.inactive_arm :
            self.stats['inactive_cnt'][-1] += 1
        elif selected_arm in self.CBQ.questions.leaves :
            self.stats['full_cnt'][-1] +=1
        else :
            self.stats['partial_cnt'][-1] +=1
        
        return selected_arm



    def answer_question(self, idx, q):
        answer, response = self.CBQ.questions.answer_question(q, idx, self.train_data)
        self.sample_ques[idx]['prev_ans'].append( response )
        self.sample_ques[idx]['prev_ques'].append( q )
        remaining_q = self.CBQ.rem_q( answer ) 
        self.sample_ques[idx]['remaining_ques'] = remaining_q
        self.sample_ques[idx]['remove_ques'] = list(set(np.arange(self.CBQ.questions.num_questions))-set(remaining_q))

        return answer, response


    def get_reward(self, i, idx, prob_mat, prob_mat_update ):
        if self.args.mode == 'bandit' :
            aig = (entropy_calc(prob_mat[i].unsqueeze(0)) - entropy_calc(prob_mat_update[i].unsqueeze(0))).item()
            aig_m = marginal_entropy_calc(prob_mat[i], self.current_labels[idx]) - marginal_entropy_calc(prob_mat_update[i], self.train_data.partial_labels[idx])          
            cosine = prob_mat[i].dot( prob_mat_update[i] ) / ( torch.linalg.norm( prob_mat[i] ) * torch.linalg.norm( prob_mat_update[i] ) )
            cosine = (2/np.pi) * np.arccos( cosine.clamp(min = -1,max = 1) )
            reward_acc = self.new_acc - self.init_acc
            reward_loss = self.init_loss - self.new_loss
            reward_loss_local = - np.log(np.dot(prob_mat[i],self.current_labels[idx])+1e-12) + np.log(np.dot(prob_mat_update[i],self.train_data.partial_labels[idx])+1e-12)
            
            reward_vect = np.array([reward_acc,
                                    reward_loss,
                                    aig,
                                    aig_m,
                                    cosine,
                                    self.local_anno_cost[i],
                                    reward_loss_local
                                    ])
            
            

            if self.args.reward == 'global':
                assert (self.lambdas[[2,3,4]]==0).all()
                reward_vect[-1] = np.sum(self.local_anno_cost)
            
            self.stats['reward_vects'].append(reward_vect)
            reward = np.sum(reward_vect * self.lambdas).item() #if self.all_arms[i] !=0 else 0.

            self.CBQ.bc = self.all_contexts[i]
            self.CBQ.update_tsp_tmp( reward, i)
        else : reward = 0.
        return reward
    
    

    def load(self,style):
        with open(os.path.join(self.temp_dir, f'{style}.pkl'), 'rb') as f:
            inputs = pickle.load(f)
        if style in ['validation', 'inputs'] :
            prob_mat = inputs['prob']
            emb_mat = inputs['emb']
            try :
                index = inputs['idx']
            except : 
                index = torch.Tensor([0])
            return prob_mat.cpu(), emb_mat.cpu(), index.cpu()
        
        elif style == 'test':
            test_acc = inputs['test_acc'].cpu().item()
            test_loss = inputs['test_loss'].cpu().item()
            self.stats['test_acc'].append(test_acc)
            self.stats['test_loss'].append(test_loss)
            print('test acc : ', test_acc, '| test loss : ', test_loss)
            return None
        
    def update_stats(self) :
        self.stats['mu_cov'].append(self.CBQ.var)
        self.stats['mu_array'] = np.concatenate((self.stats['mu_array'], self.CBQ.mu_cap),axis=1)
        self.stats['label_type_cnt']['FL'].append( len( self.train_data.classified_indices() ) )
        self.stats['label_type_cnt']['PL'].append( len(list( (set(self.train_data.train_indices()) - set(self.train_data.classified_indices())) ) ) )
        self.stats['label_type_cnt']['UL'].append(  len(self.train_set.indices) - len(self.train_data.train_indices())   )
        self.stats['contexts'].append(self.all_contexts)
        self.init_loss = self.new_loss
        self.init_acc = self.new_acc
        
    
    def save_stats(self):
        data_dir = self.args.save_repo
        if not os.path.exists(data_dir) : os.makedirs(data_dir) 
        file_name = os.path.join(data_dir, self.file_name)
        with open(file_name, 'wb') as f:
            pickle.dump(self.stats, f)

    

    def work(self):
        pass