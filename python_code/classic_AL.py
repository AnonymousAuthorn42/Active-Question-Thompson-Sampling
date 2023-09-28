import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import Subset
from al_utils import AL_process
from questions import entropy_calc
from config import *





class ClassicAL_process(AL_process) :
    '''
    Need to be instentiated with args.mode in ['entropy', 'least confidence', 'random']
    '''
    def __init__(self, args):
        super(ClassicAL_process, self).__init__(args)
        self.args = args
        
        self.annotation_cost = np.unique(self.CBQ.questions.costs).sum() #self.args.leaf_cost
        self.eval_val = False

        
        self.file_name = f'{self.args.dataset_name}_{self.args.architecture}_{self.args.mode}-sampling.pkl'

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
        self.train_data.true_labels(index)
        self.stats['label_type_cnt']['FL'].append(self.stats['label_type_cnt']['FL'][-1] + len(index) )
        self.stats['label_type_cnt']['UL'].append(self.stats['label_type_cnt']['UL'][-1] - len(index))
        self.stats['label_type_cnt']['PL'].append(0)
        self.stats['anno_cnt'].append(len(index)*self.annotation_cost)

    

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
            main.file_name =  f'{args.dataset_name}_{args.architecture}_{args.mode}-sampling_{repetition}.pkl'
        main.work()
        main.clear_temp_path()
        my_seed +=1
    print(datetime.datetime.now()-now)