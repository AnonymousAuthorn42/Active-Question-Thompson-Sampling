import datetime
import numpy as np
import torch
from torch.distributed import destroy_process_group
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from networks import  Trainer
from al_utils import *
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


class AQTS_process(AL_process):
    def __init__(self,args) :
        super(AQTS_process,self).__init__(args)
        self.args = args
        self.budget = len(self.train_set)*self.args.leaf_cost*self.args.budget
        self.eval_val = True

        if args.mode =='eig' : assert args.contexts[0]=='eig'

        self.eval_val = True # needed to compute the rewards
        algo_name = 'AQTS-FA'
        self.file_name = f'{self.args.dataset_name}_{self.args.architecture}_{algo_name}.pkl'
        self.train_file_name = 'training_stats.pkl'

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

    
    def get_question(self, i, idx, prob_mat, emb_mat, pr_val, emb_val) :
        self.CBQ.sample_mu()
                
        y_partial = self.train_data.partial_labels[idx].cpu()
        partials = self.train_data.partial_labels[self.train_set.indices].cpu()
        
        Y_val = self.train_data.partial_labels[self.val_set.indices].cpu()

        question_sample_context = self.CBQ.questions.context_generation(prob_mat[i], y_partial, partials,                                                                            
                                                                    pr_val, Y_val, emb_val, emb_mat[i] )

        if self.args.inactive_zero_context :
            question_sample_context[:,0] = 0
        # SELECT QUESTION - BANDIT PART HERE
        selected_arm = self.CBQ.choose_arm(question_sample_context, self.sample_ques, idx, 
                                            self.args.mode, self.args.all_arms_available)
        
        self.all_contexts.append(question_sample_context[:,selected_arm])
        self.all_arms.append(selected_arm)
        self.local_anno_cost.append(self.CBQ.questions.costs[selected_arm])
        self.stats['anno_cnt'][-1] += self.CBQ.questions.costs[selected_arm]
        
        if selected_arm ==0 and self.args.inactive_arm :
            self.stats['inactive_cnt'][-1] += 1
        elif selected_arm in self.CBQ.questions.leaves :
            self.stats['full_cnt'][-1] +=1
        else :
            self.stats['partial_cnt'][-1] +=1

        if np.sum(self.stats['anno_cnt']) >= self.budget :
            return selected_arm, False
        else :
            return selected_arm, True
        
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
            while update_cnt < self.args.num_pts :
                idxs = self.get_batch_example(batch_size = self.args.num_pts//10)
                prob, emb = all_pr[idxs], all_emb[idxs]                    
                for i, idx in enumerate(idxs) :
                    question, continu = self.get_question(i, idx, prob, emb, pr_val, emb_val )
                    _, _ = self.answer_question(idx, question)
                    if not self.args.inactive_arm or question != 0 :
                        update_cnt +=1
                    batch_size +=1
                    if not continu :
                        idxs =idxs[:i+1]
                        break
                rnd_shuffle += idxs.tolist()
                if batch_size <= self.args.num_pts//10 :
                    prob_mat = prob[:i+1]
                    emb_mat = emb[:i+1]
                else :
                    prob_mat = torch.vstack((prob_mat,prob[:i+1]))
                    emb_mat = torch.vstack((emb_mat,emb[:i+1]))
                if batch_size > 10*self.args.num_pts or not continu :
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
            assert(len(prob_mat_update) == len(prob_mat))

            for i, idx in enumerate(rnd_shuffle):
                _ = self.get_reward(i, idx, prob_mat, prob_mat_update)
            self.CBQ.update_thompson_sampling()
            unlab_this_rnd = np.sum([1 for idx in rnd_shuffle if self.train_data.partial_labels[idx].sum().item()== self.train_data.num_classes])
            print('unlabeled :', unlab_this_rnd)
            print(np.bincount(self.all_arms))
            
            self.update_stats()
            if self.args.save : self.save_stats()
            if not continu :
                break
        
        #-------- Final training -----------------------
        self.args.lr = 0.001 # PLL need small lr even for the first training epochs
        self.reload_datasets()
        self.define_model(net_name = self.args.architecture)
        self.final_training(world_size)
            
        
                



if __name__ =='__main__' :
    args = parse_args()
    my_seed = args.seed
    args.allow_multichoice = False
    args.explore_all = True
    args.pool_is_X = False

    
    now = datetime.datetime.now()
    for repetition in range(args.num_rep):
        rep_now = datetime.datetime.now()
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        print('repetition', repetition)
        main = AQTS_process(args)
        if args.num_rep >1 :
            main.file_name =  f'{args.dataset_name}_{args.architecture}_TrainOn_{repetition}.pkl'
            main.train_file_name = f'training_stats_{repetition}.pkl'
        main.work()
        #main.clear_temp_path()
        my_seed +=1
        print( 'rep time', datetime.datetime.now() - rep_now )

    print('total time', datetime.datetime.now()-now)


