import datetime
import numpy as np
import torch
from torch.distributed import destroy_process_group
from networks import  Trainer
from al_utils import *
from config import *


class AQTS_process(AL_process):
    def __init__(self,args) :
        super(AQTS_process,self).__init__(args)
        self.args = args
        
        self.eval_val = True

        if args.mode =='eig' : assert args.contexts[0]=='eig'

        self.eval_val = True # needed to compute the rewards
        algo_name = 'AQTS-FA'
        self.file_name = f'{self.args.dataset_name}_{self.args.architecture}_{algo_name}.pkl'

    def train_model(self, max_epochs):
        super().train_model(max_epochs)
        input_loader = self.define_loader(self.train_data, train = False) 
        self.get_prob_and_emb(input_loader, style = 'inputs')

    def multigpu_train(self, rank, world_size, n_epoch) :
        ddp_setup(rank, world_size)
        test_loader = self.define_loader( dataset = self.test_data, train = False)
        loader = self.define_loader()
        self.net.train()
        trainer = Trainer(self.net, loader, self.optimizer, self.loss, self.temp_dir, rank)
        trainer.train(n_epoch)
        trainer.eval(test_loader, 'test')
        input_loader = self.define_loader(self.train_data, train = False)
        trainer.eval(input_loader,'inputs')
        if self.eval_val :
            val_loader = self.define_loader(dataset = self.val_set, train = False)
            trainer.eval(val_loader, 'validation')
        if rank == 0 : 
            torch.save(trainer.model.module.state_dict(), os.path.join(self.temp_dir, 'current_model.pt') ) 
        destroy_process_group()
            
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
            all_pr, all_emb, index = self.load('inputs')
            reorder = argsort_as(index, np.arange(len(self.train_data)))
            all_pr = all_pr[reorder]
            all_emb = all_emb[reorder]
            pr_val, emb_val = all_pr[self.val_set.indices], all_emb[self.val_set.indices]
            update_cnt = 0
            batch_size = 0
            rnd_shuffle = []
            
            self.init_answering_process()
            self.net.eval()
            while update_cnt < self.args.num_pts and len(self.train_data.remaining_samples()) >0 :
                idxs = self.get_batch_example(batch_size = self.args.num_pts//10)
                rnd_shuffle += idxs.tolist()
                prob, emb = all_pr[idxs], all_emb[idxs]                    
                if update_cnt == 0 :
                    prob_mat = prob
                    emb_mat = emb
                else :
                    prob_mat = torch.vstack((prob_mat,prob))
                    emb_mat = torch.vstack((emb_mat,emb))
                for i, idx in enumerate(idxs) :
                    question = self.get_question(i, idx, prob, emb, pr_val, emb_val )
                    _, _ = self.answer_question(idx, question)
                    if not self.args.inactive_arm or question != 0 :
                        update_cnt +=1
                    batch_size +=1
                if batch_size > 10*self.args.num_pts :
                    break
            print('updated points :', update_cnt)
            print('annotation batch size :', batch_size)
            
            now = datetime.datetime.now()
            self.define_model()
            self.train(world_size)
            print('training time',datetime.datetime.now()-now, 'current data size', len(self.train_data.train_indices()))
            
            self.new_acc, self.new_loss = self.get_performances()
            self.load('test')
            #input_set = Subset(self.train_data,rnd_shuffle)
            prob_mat_update,_,index = self.load('inputs') #self.compute_prob_and_emb(world_size, input_set, rnd_shuffle, style = 'inputs')
            reorder = argsort_as(index, np.arange(len(self.train_data)))
            prob_mat_update = prob_mat_update[reorder]
            prob_mat_update = prob_mat_update[rnd_shuffle]

            for i, idx in enumerate(rnd_shuffle):
                _ = self.get_reward(i, idx, prob_mat, prob_mat_update)
            if self.args.mode == 'bandit' : self.CBQ.update_thompson_sampling()
            unlab_this_rnd = np.sum([1 for idx in rnd_shuffle if self.train_data.partial_labels[idx].sum().item()== self.train_data.num_classes])
            print('unlabeled :', unlab_this_rnd)
            print(np.bincount(self.all_arms, minlength=self.CBQ.num_arms))

            self.update_stats()
            if self.args.save : self.save_stats()
            if len(self.train_data.remaining_samples()) == 0 : break
                



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
            main.file_name =  f'{args.dataset_name}_{args.architecture}_AQTS-FA_{repetition}.pkl'
        main.work()
        main.clear_temp_path()
        my_seed +=1
        print( 'rep time', datetime.datetime.now() - rep_now )

    print('total time', datetime.datetime.now()-now)


