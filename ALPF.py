import datetime
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.utils.data import  Subset
from al_utils import AL_process
from config import *



class ALPF_process(AL_process):
    def __init__(self, args):
        super(ALPF_process,self).__init__(args)
        self.args = args
        self.Q = self.CBQ.questions.Q
        self.stats = {'anno_cnt': [],
                    'partial_cnt' : [],
                    'full_cnt' : [],
                    'inactive_cnt' :[],
                    'test_acc' : [],
                    'test_loss' :[],
                    'label_type_cnt' : {'FL' : [0],
                                        'PL' : [0],
                                        'UL' : [len(self.train_set.indices)]},
                    'config' : args
                    }
        
        self.eval_val = False
        self.file_name = f'{args.dataset_name}_{args.architecture}_ALPF.pkl'

    def eig(self, pr):
        qp = np.dot(pr, self.Q.T)
        qp = np.clip(qp,a_min=1e-12,a_max=1.-1e-12)
        mere = np.nan_to_num(-(qp*np.log(qp)+(1-qp)*np.log(1-qp))) 
        return mere
    
    def get_batch_examples(self, prob_mat, index):
        eigs = self.eig(prob_mat)
        quests = eigs.argmax(axis=1)
        idx_sample = eigs.max(axis=1).argsort()[-self.args.num_pts:] 
        rnd_shuffle = index[idx_sample].numpy()
        questions = quests[idx_sample]
        self.stats['anno_cnt'][-1] += self.CBQ.questions.costs[questions].sum()
        for q in questions : 
            if q ==0 and self.args.inactive_arm :
                self.stats['inactive_cnt'][-1] += 1
            elif q in self.CBQ.questions.leaves :
                self.stats['full_cnt'][-1] +=1
            else :
                self.stats['partial_cnt'][-1] +=1
        return rnd_shuffle, questions

    def init_answering_process(self):
         self.stats['partial_cnt'].append(0)
         self.stats['full_cnt'].append(0)
         self.stats['inactive_cnt'].append(0)
         self.stats['anno_cnt'].append(0)

    def update_stats(self) :
        self.stats['label_type_cnt']['FL'].append( len( self.train_data.classified_indices() ) )
        self.stats['label_type_cnt']['PL'].append( len(list( (set(self.train_data.train_indices()) - set(self.train_data.classified_indices())) ) ) )
        self.stats['label_type_cnt']['UL'].append(  len(self.train_set.indices) - len(self.train_data.train_indices())   )

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

    def work(self):
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
            input_set = Subset( self.train_data, self.train_data.remaining_samples() )
            prob_mat, index = self.compute_prob_and_emb(world_size, input_set, style='inputs')

            self.init_answering_process()

            rnd_shuffle, questions = self.get_batch_examples(prob_mat, index)

            for idx, q in zip(rnd_shuffle, questions):
                answer, response = self.answer_question(idx, q)

            self.define_model()
            self.train(world_size)
            self.load('test')

            self.update_stats()
            if self.args.save : self.save_stats()
            if len(self.train_data.remaining_samples()) ==0:
                print('dataset completely labeled at rnd', rnd+1)
                break



if __name__ =='__main__' :
    args = parse_args()
    my_seed = args.seed
    now = datetime.datetime.now()

    for repetition in range(args.num_rep):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        print('repetition', repetition)
        main = ALPF_process(args)
        if args.num_rep >1 :
            main.file_name =  f'{args.dataset_name}_{args.architecture}_ALPF_{repetition}.pkl'
        main.work()
        main.clear_temp_path()
        my_seed +=1
    print(datetime.datetime.now()-now)

