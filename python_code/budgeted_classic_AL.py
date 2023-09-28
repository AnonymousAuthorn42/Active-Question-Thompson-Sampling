import os
import datetime
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset
from al_utils import AL_process, ddp_setup
from networks import Trainer
from torch.distributed import destroy_process_group
import torch.distributed as dist
from datasets import preprocess_data, MyDataset
from torch.optim.lr_scheduler import MultiStepLR
from questions import entropy_calc
from config import *


class TTrainer(Trainer) :
    def __init__(self, model, dataloader, test_loader, optimizer, loss, data_dir, gpu_id) :
        super(TTrainer, self).__init__(model, dataloader, optimizer, loss, data_dir, gpu_id)

        self.test_loader = test_loader
        self.stats ={'test_acc' : [],
                    'test_loss' : []}
        self.lr_scheduler = MultiStepLR(self.optimizer,[100,130], gamma =0.1)
        self.file_name = 'training_stat.pkl'
    def run_epoch(self, epoch) :
        self.loader.sampler.set_epoch(epoch)
        for inputs, targets, _ in self.loader :
            inputs = inputs.to(self.gpu_id, non_blocking = True)
            targets = targets.to(self.gpu_id, non_blocking = True)
            self.run_batch(inputs, targets)

    def train(self, max_epochs):
        
        self.model.train()
        for epoch in range(max_epochs):
            if self.gpu_id == 0 :
                print('epoch : ', epoch +1) 
            self.run_epoch(epoch)
            self.lr_scheduler.step()
            if (epoch +1) % 10 == 0 :
                self.eval(self.test_loader)
                self.model.train()
                

    def eval(self, loader) :
        self.model.eval()
        val_loss = torch.Tensor([0.]).to(self.gpu_id)
        val_acc = torch.Tensor([0.]).to(self.gpu_id)
        with torch.no_grad():
            for inputs, targets,_ in loader :
                output, _ = self.model( inputs.to(self.gpu_id, non_blocking = True) )
                targets = targets.to(self.gpu_id, non_blocking = True)
                loss = self.loss(output, targets)
                val_loss += loss * len(inputs) / (len(loader.dataset))
                val_acc += (output.argmax(-1) == targets.argmax(-1)).sum() / len(loader.dataset)        
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        if self.gpu_id == 0 :
            self.stats['test_acc'].append(val_acc.item())
            self.stats['test_loss'].append(val_loss.item())
            print(self.stats)
            with open(os.path.join(self.data_dir,self.file_name),'wb') as f:
                pickle.dump(self.stats, f)



class ClassicAL_process(AL_process) :
    '''
    Need to be instentiated with args.mode in ['entropy', 'least confidence', 'random']
    '''
    def __init__(self, args):
        super(ClassicAL_process, self).__init__(args)
        self.args = args
        
        self.annotation_cost = np.unique(self.CBQ.questions.costs).sum()#self.args.leaf_cost
        self.budget = len(self.train_set)*self.args.leaf_cost*self.args.budget
        self.eval_val = False

        
        self.file_name = f'{self.args.dataset_name}_{self.args.architecture}_{self.args.mode}-sampling.pkl'
        self.train_file_name = 'training_stats.pkl'
        self.stats = {'anno_cnt': [],                   
                        'full_cnt' : [],
                        'test_acc' : [],
                        'test_loss' :[],
                        'reward_vects' :[],
                        'label_type_cnt' : {'FL' : [0],
                                            'PL' : [0],
                                            'UL' : [len(self.train_set.indices)]},
                        'config' : args
                    }

    def compute_prob_and_emb(self, world_size, dataset, style='inputs'):
        if world_size > 1 and self.args.device != 'cpu' : 
            mp.spawn(self.get_prob_and_emb_multi, 
                        args = (world_size, dataset, style, ), 
                        nprocs = world_size)
            prob_mat, _, input_idx = self.load(style)
        else : 
            self.net.to(self.device)
            input_loader = self.define_loader(dataset = dataset, train = False)
            self.get_prob_and_emb(input_loader, style)
            prob_mat, _, input_idx = self.load(style)
        return prob_mat, input_idx
        

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

    def get_batch_example(self, prob_mat = None, index = None):
        if self.args.mode == 'random' :
            examples_idx = np.random.permutation(self.train_data.remaining_samples())[:self.args.num_pts]  
        else :
            assert prob_mat is not None and index is not None
            if self.args.mode == 'entropy' :
                entropies = entropy_calc(prob_mat)
                idx = np.argsort(entropies)[-self.args.num_pts:]
            elif self.args.mode == 'least confidence' :
                uncertainties =(1-prob_mat.max(-1))
                idx = np.argsort(uncertainties)[-self.args.num_pts:]
            examples_idx = index[idx]
            
        self.stats['full_cnt'].append(len(examples_idx))
        return examples_idx

    def label_examples(self, index):
        self.stats['anno_cnt'].append(0)
        c = 0
        for idx in index :
            self.train_data.true_labels(idx)
            c+=1
            self.stats['anno_cnt'][-1] += self.annotation_cost
            if np.sum(self.stats['anno_cnt']) >= self.budget :
                break
        
        self.stats['label_type_cnt']['FL'].append(self.stats['label_type_cnt']['FL'][-1] + c )
        self.stats['label_type_cnt']['UL'].append(self.stats['label_type_cnt']['UL'][-1] - c)
        self.stats['label_type_cnt']['PL'].append(0)
        
    def final_training_gpu(self, rank, world_size, n_epoch) :
        ddp_setup(rank, world_size)
        test_loader = self.define_loader( dataset = self.test_data, train = False)
        loader = self.define_loader()
        self.net.train()
        trainer = TTrainer(self.net, loader, test_loader, self.optimizer, self.loss, self.args.save_repo, rank)
        trainer.file_name = self.train_file_name
        trainer.train(n_epoch)
        if rank == 0 : 
            torch.save(trainer.model.module.state_dict(), os.path.join(self.temp_dir, 'current_model.pt') )
        destroy_process_group()

    def final_training(self, world_size ) :
        mp.spawn(self.final_training_gpu, args = (world_size, 150,), nprocs = world_size)
        state = torch.load(os.path.join(self.temp_dir, 'current_model.pt') )
        self.net.load_state_dict(state)
    
    def reload_datasets(self):
        preprocess = preprocess_data(args.architecture, args.resnet_type, args.dataset_name)
        partial_labels = self.train_data.partial_labels.clone()
        self.train_data = MyDataset(args.dataset_name, args.data_dir, train_bool = True, preprocess = preprocess)
        self.test_data = MyDataset(args.dataset_name, args.data_dir, train_bool = False, preprocess = preprocess)
        self.train_data.set_val_indices(self.val_set)
        self.test_data.true_labels()
        self.train_data.partial_labels = partial_labels

    def work(self) :
        world_size = torch.cuda.device_count()
        print('architecture :', self.args.architecture)
        print('dataset :', self.args.dataset_name)
        
        self.initialize_process()
        self.define_model()
        if len(self.train_data.classified_indices() ) > 0 :
            self.train(world_size)
        self.load('test')

        for rnd in range(self.args.num_rounds) :
            print('round ', rnd+1)
            if len(self.train_data.remaining_samples()) == 0 :
                break
            if self.args.mode != 'random':
                input_set = Subset( self.train_data, self.train_data.remaining_samples() )
                prob_mat, input_idx = self.compute_prob_and_emb(world_size, input_set, style = 'inputs')
            else :
                prob_mat, input_idx = None, None
            

            rnd_shuffle = self.get_batch_example(prob_mat, input_idx)
            
            self.label_examples(rnd_shuffle)

            self.define_model()
            self.train(world_size)        
            self.load('test')

            
            if self.args.save : self.save_stats()
            if np.sum(self.stats['anno_cnt']) >= self.budget :
                break
        self.args.lr = 0.001 # PLL need small lr even for the first training epochs
        self.reload_datasets()
        self.define_model(net_name = self.args.architecture)
        self.final_training(world_size)


if __name__ =='__main__' :
    args = parse_args()
    my_seed = args.seed

    now = datetime.datetime.now()
    for repetition in range(args.num_rep):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        print('repetition', repetition)
        main = ClassicAL_process(args)
        if args.num_rep >1 :
            main.file_name =  f'{args.dataset_name}_{args.architecture}_TrainOn-{args.mode}_{repetition}.pkl'
            main.train_file_name = f'training_stats_{repetition}.pkl'
        main.work()
        main.clear_temp_path()
        my_seed +=1
    print(datetime.datetime.now()-now)