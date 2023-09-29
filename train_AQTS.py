import datetime
import numpy as np
import torch
from torch.utils.data import Subset
from al_utils import *
from config import *


class AQTS_process(AL_process):
    def __init__(self,args) :
        super(AQTS_process,self).__init__(args)
        self.args = args
        
        self.eval_val = True

        if args.mode =='eig' : assert args.contexts[0]=='eig'

    def work(self) :
            world_size = torch.cuda.device_count()
            print('num gpus availables :', world_size)
            print('device :' , self.args.device)
            print('architecture :', self.args.architecture)
            print('dataset :', self.args.dataset_name)

            self.initialize_process()
            self.define_model()
            

            if len(self.train_data.classified_indices() ) > 0 :
                self.train(world_size)
            self.init_acc, self.init_loss = self.get_performances() 
            self.load('test')
            
            for rnd in range(self.args.num_rounds) :
                print('round ', rnd+1)
                rnd_shuffle = self.get_batch_example()
                input_set = Subset(self.train_data,rnd_shuffle)
                
                prob_mat, emb_mat = self.compute_prob_and_emb(world_size, input_set, rnd_shuffle, style = 'inputs')
                pr_val, emb_val, _ = self.load('validation')
                
                self.init_answering_process()
                for i, idx in enumerate(rnd_shuffle):
                    question = self.get_question( i, idx, prob_mat, emb_mat, pr_val, emb_val )
                    _, _ = self.answer_question(idx, question)
                
                now = datetime.datetime.now()
                self.define_model()
                self.train(world_size)
                print('training time',datetime.datetime.now()-now, 'data size', len(self.train_data.train_indices()))
                
                self.new_acc, self.new_loss = self.get_performances()
                self.load('test')

                prob_mat_update, _ = self.compute_prob_and_emb(world_size, input_set, rnd_shuffle, style = 'inputs')


                for i, idx in enumerate(rnd_shuffle):
                    _ = self.get_reward(i, idx, prob_mat, prob_mat_update)
                self.CBQ.update_thompson_sampling()
                print('unlabeled :', np.sum([1 for idx in rnd_shuffle if self.train_data.partial_labels[idx].sum().item()== self.train_data.num_classes]))
                print(np.bincount(self.all_arms))

                self.update_stats()
                if self.args.save : self.save_stats()
                



if __name__ =='__main__' :
    args = parse_args()
    my_seed = args.seed
    
    now = datetime.datetime.now()
    for repetition in range(args.num_rep):
        rep_now = datetime.datetime.now()
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        print('repetition', repetition)
        main = AQTS_process(args)
        if args.num_rep >1 :
            main.file_name =  main.file_name[:-4] + f'_{repetition}.pkl'
        main.work()
        main.clear_temp_path()
        my_seed +=1
        print( 'rep time', datetime.datetime.now() - rep_now )

    print('total time', datetime.datetime.now()-now)


