import os
import datetime
import numpy as np
import torch
from torch.utils.data import Subset
from al_utils import *
from config import *

class AQTS_process(AL_process):
    def __init__(self, args):
        super(AQTS_process, self).__init__(args)
        self.temp_dir = 'temp/temp4'
        if not os.path.exists(self.temp_dir):
            os.mkdir(self.temp_dir)

        self.eval_val = True

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
        num_unlabeled = self.stats['label_type_cnt']['UL'][0]
        self.available_arms = np.ones( (num_unlabeled,self.CBQ.num_arms) )
        self.global_index = self.train_data.remaining_samples()
        self.contexts = torch.zeros( (num_unlabeled,self.CBQ.context_dim ,self.CBQ.num_arms) )
        

    def update_contexts(self, prob_mat, emb_mat, pr_val, emb_val):
        Y_partial = self.train_data.partial_labels[self.global_index]
        self.contexts = self.CBQ.questions.context_generation(prob_mat, Y_partial, Y_partial, 
                                                              self.train_data.labels[self.val_set.indices], pr_val, emb_val, emb_mat)
        
    def choose_arm(self) :
        
        self.CBQ.sample_mu()
        
        # SELECT BEST ARM - BANDIT PART HERE
        pred = (self.contexts.transpose((0,2,1)) @ self.CBQ.mu_bar) * self.available_arms
        idx = np.nanargmax(pred)
        rnd_samp = idx // self.CBQ.num_arms
        question =  idx % self.CBQ.num_arms
        global_idx = self.global_index[rnd_samp]

        self.local_anno_cost.append(self.CBQ.questions.cost_annotation()[question])
        self.stats['anno_cnt'][-1] += self.CBQ.questions.cost_annotation()[question]
        if question ==0 and self.args.inactive_arm :
            self.stats['inactive_cnt'][-1] += 1
        elif question in self.CBQ.questions.leaves :
            self.stats['full_cnt'][-1] +=1
        else :
            self.stats['partial_cnt'][-1] +=1

        return rnd_samp, global_idx, question
    
    def answer_question(self, idx, global_idx, question) :
        answer, response = self.CBQ.questions.answer_question(question, global_idx, self.train_data)
        remaining_q = self.CBQ.rem_q( answer ) 
        if not self.args.all_arms_available :
            self.available_arms[idx] = np.array([1. if (k in remaining_q) else np.nan for k in range(self.CBQ.num_arms)])
        self.all_contexts.append( self.contexts[idx,:, question] )
        self.all_arms.append(question)
        self.batch_idx.append(global_idx)

    def init_answering_process(self):
        super().init_answering_process()
        self.batch_idx = []

    def get_rewards(self, prob_mat, prob_mat_update) :
        for i, global_idx in enumerate(self.batch_idx) :
            question = self.all_arms[i]
            reward = self.get_reward(i, global_idx, question, prob_mat, prob_mat_update)

    def work(self):
        world_size = torch.cuda.device_count()
        print('architecture :', self.args.architecture)
        print('dataset :', self.args.dataset_name)

        self.initialize_process()
        self.define_model()
        if len(self.train_data.classified_indices() ) > 0 :
            self.train(world_size)
        
        self.init_acc, self.init_loss = self.get_performances() 
        self.load('test')
    
        input_set = Subset(self.train_data,self.global_index)
    
        annotation_batch_size = 0
        for rnd in range(self.args.num_rounds*self.args.num_pts) :
            if annotation_batch_size == 0 :
                prob_mat, emb_mat = self.compute_prob_and_emb(world_size, input_set, self.global_index, style = 'inputs')
                self.init_answering_process()
                assert len(self.all_contexts) == 0
                assert len(self.batch_idx) == 0
                pr_val, emb_val, _ = self.load('validation')

                self.update_contexts(prob_mat, emb_mat, pr_val, emb_val)
                
            idx, global_idx, question = self.choose_arm()
            self.answer_question(idx, global_idx, question)
            annotation_batch_size +=1

            if annotation_batch_size == self.args.num_pts :
                self.define_model()
                self.train(world_size)
                self.load('test')

                self.new_acc, self.new_loss = self.get_performances()
                prob_mat_update, emb_mat_update = self.compute_prob_and_emb(world_size, input_set, self.global_index, style = 'inputs')

                self.get_rewards( prob_mat, prob_mat_update)
                self.CBQ.update_thompson_sampling()

                annotation_batch_size = 0
                self.update_stats()
                if self.args.save : self.save_stats()


if __name__ =='__main__' :
    args = parse_args()
    my_seed = args.seed

    now = datetime.datetime.now()
    for repetition in range(args.num_rep):
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        print('repetition', repetition)
        main = AQTS_process(args)
        if args.num_rep >1 :
            main.file_name =  f'{args.dataset_name}_{args.architecture}_{args.mode}_{repetition}.pkl'
        main.work()
        my_seed +=1

    print(datetime.datetime.now()-now)