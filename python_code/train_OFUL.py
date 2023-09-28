import os
import pickle
import datetime
import numpy as np
np.random.seed(21)
from sklearn.model_selection import train_test_split

import torch
torch.manual_seed(21)
import torch.optim as optim

from datasets import MyDataset, ToyDataset, preprocess_data
from networks import train_partial, EarlyStopper, mlpMod, test_partial, get_embedding, get_prob_and_emb, weight_reset
from al_utils import ContextualBanditsQuestions, rem_q
from losses import *
from questions import entropy_calc, Questions
from config import *



now = datetime.datetime.now()
device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
device = 'cpu'

# --------------------------------------------------------------------------------------------------
#                                           GET DATA
# --------------------------------------------------------------------------------------------------
print('Loading data')
training_data = MyDataset(dataset_name, data_dir, train_bool=True, num_classes = circles_num_classes)
test_data = MyDataset(dataset_name, data_dir, train_bool=False, num_classes = circles_num_classes)
class_to_idx = training_data.class_to_idx

num_training_samples = len(training_data)
num_test_samples = len(test_data)
num_classes = training_data.num_classes
img_dim = training_data.img_dim

if net_name == 'resnet':
    training_file = f'{dataset_name}_train_resnet{resnet_type}.pkl'
    testing_file = f'{dataset_name}_test_resnet{resnet_type}.pkl'
    if training_file in os.listdir(data_dir) :
        with open(os.path.join(data_dir,training_file),'rb') as f:
            training_data = pickle.load(f)
        with open(os.path.join(data_dir,testing_file), 'rb') as f:
            test_data = pickle.load(f)
        X_all = training_data['data']
        Y_all = training_data['label']
        X_test = test_data['data']
        Y_test = test_data['label']
        img_dim = X_all[0].shape          
    else :
        print('preprocessing data - please wait ± 1 hour ')
        X_all, Y_all, X_test, Y_test = preprocess_data(dataset_name, data_dir, num_classes, resnet_type)
        img_dim = X_all[0].shape
        print('preprocessing done - saving it in your data_dir')
        training_data = {'data' : X_all, 'label' : Y_all}
        test_data = {'data' : X_test, 'label' : Y_test}
        with open(os.path.join(data_dir, training_file),'wb') as f :
            pickle.dump(training_data,f)
        with open( os.path.join(data_dir, testing_file), 'wb' ) as f :
            pickle.dump(test_data,f)
    # memory control
    del training_data, test_data, training_file, testing_file
        
       
else :
    X_all = torch.zeros((num_training_samples, *img_dim))
    Y_all = torch.zeros((num_training_samples, num_classes))
    X_test = torch.zeros((num_test_samples, *img_dim))
    Y_test = torch.zeros((num_test_samples, num_classes))
    for sample in range(num_training_samples):
         X, y, i = training_data.__getitem__(sample)
         X_all[i] = X
         Y_all[i, y] = 1
    for sample in range(num_test_samples):
         X, y, i = test_data.__getitem__(sample)
         X_test[i] = X
         Y_test[i, y] = 1

if init_lab_prop:
    init_lab_size = int(X_all.shape[0]*init_lab_prop)

X, X_oth, Y, Y_oth = train_test_split(X_all, Y_all,
                                          test_size=1-unlab_set_prop, random_state=42)
#X_l, Y_l, X_val, Y_val = X_oth[0:init_lab_size], Y_oth[0:init_lab_size], X_oth[init_lab_size:2*init_lab_size], Y_oth[
#                                                                                                    init_lab_size:2*init_lab_size]
X_l, Y_l, X_val, Y_val = X_oth[0:init_lab_size], Y_oth[0:init_lab_size], X_oth[init_lab_size:], Y_oth[
                                                                                                    init_lab_size:]
X = X[0:unlabeled_subset_size, :]
Y = Y[0:unlabeled_subset_size, :]
# --------------------------------------------------------------------------------------------------
#                                           INIT BANDIT
# --------------------------------------------------------------------------------------------------
print('Initializing bandit')
#GET QUESTIONS AND INIT QUESTION BANDIT
FMQ = Questions(dataset_name, hyp_emb_file, label_to_idx = class_to_idx, context_type=context_type, root = inactive_arm)
CBQ = ContextualBanditsQuestions( nu2, questions = FMQ, gamma_forget = gamma_forget, lambda_regul = lambda_regul)

num_unlabeled = X.shape[0]
sample_label_state = {'FL': [], 'PL': [], 'UL': list(np.arange(num_unlabeled))}


Y_partial = torch.ones((num_unlabeled, num_classes))
sample_ques = {i: {'remaining_ques': np.arange(CBQ.questions.num_questions), 'prev_ques':[], 'prev_ans':[], 'remove_ques': []} for i in range(num_unlabeled)}
available_arms = np.ones((num_unlabeled,CBQ.questions.num_questions))
contexts = np.zeros((num_unlabeled,CBQ.context_dim ,CBQ.questions.num_questions))


# Keep count of question, annotations, and approx - partial and full questions
ques_cnt = [0]
anno_cnt = [0]
pos_answer_cnt =[0] 
partial_qtype_cnt = [0]
full_qtype_cnt = [0]
Y_partial_state = {10 : [num_unlabeled],
                   **{k : [0] for k in range(1,10)}}
root_selection = [0]
mu_array = CBQ.mu_cap
reward_pred =[]
reward_real = []

# --------------------------------------------------------------------------------------------------
#                                           SETUP NETWORK
# --------------------------------------------------------------------------------------------------

embSize = 256 if num_classes == 10 else 1024

if net_name == 'mlp' or net_name == 'resnet' :
    net = mlpMod(dim=img_dim, nclasses=num_classes, embSize = embSize)
    net.to(device = device)


        
if net_name =='lenet':
    grayscale = (dataset_name=='FashionMNIST')
    net = LeNet5(grayscale = grayscale)
    net.to(device)

    

    
criterion = { 'partial_loss' : partial_loss,
             'hierloss' : HierarchicalCrossEntropyLoss(CBQ, device = device),
             'partial_hierloss' : Partial_HierLoss(CBQ, device = device),
             'composite_loss' : Composite_loss(CBQ, device = device),
             'focal_PHXE': Focal_PHXE(CBQ, tau = tau, device = device)}[loss_type]

if loss_weights is not None :
    if 'hierloss' in loss_type or loss_type == 'focal_PHXE' :
        criterion.new_weights(loss_weights)

if optimizer_type=='adam' :
        optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999)) 
elif optimizer_type=='SGD':
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)          
           
    
# train on init labeled set and get acc
print('initial training')
net.apply(weight_reset)
stopping = EarlyStopper(patience=3,min_delta=0.5 ) if early_stopping else None
train_partial(X_l, Y_l, net, optimizer, criterion, 30, batch_size, device)
init_loss , init_acc = test_partial(X_val, Y_val, net, criterion, batch_size, device)
loss , acc = test_partial(X_test, Y_test, net, criterion, batch_size, device)
test_acc = [acc]
test_loss = [loss]
print('initial acc :', acc)
# --------------------------------------------------------------------------------------------------
#                                           START ROUNDS
# --------------------------------------------------------------------------------------------------
batch = 0   
    
for r in range(num_rounds):
    if (r+1)%10 == 0 :
        print('round #', r+1)
    if batch ==0:
        # INIT TRACKING LISTS
        all_arms = []
        all_contexts = []
        updated_points = []
        pos_answers = 0
        num_root = 0

    
    # sample selection strategy
    if batch == 0 :
        print("computing contexts")
        temp = datetime.datetime.now()
        if pool_is_X :
            remaining_samples =list(np.arange(num_unlabeled))
        else :
            remaining_samples = list(set(np.arange(num_unlabeled))-set(sample_label_state['FL']))

        input_index = {samp : i for i,samp in enumerate(remaining_samples)}
        inputs = X[remaining_samples]
        prob_mat, emb_mat = get_prob_and_emb(inputs, net, 100, device, dim = net.embSize)
    
    
        # get embeddings for hyperbolic stuff that do not change over selections in this round of labeling
        if len(sample_label_state['FL']) > 0:
            X_labeled = torch.cat([X_l, X[sample_label_state['FL']]])
            label_l = torch.cat([Y_l, Y[sample_label_state['FL']]]).numpy()
            emb_l = get_embedding(X_labeled, net, 100, device, dim=net.embSize)
        else:
            label_l = Y_l.numpy()
            emb_l = get_embedding(X_l, net, 100,  device, dim=net.embSize)
        



        #COMPUTE ALL CONTEXTS
        
        norms =np.stack([np.linalg.norm(emb_mat[i] - emb_l, axis=1) for i in range(emb_mat.shape[0])])
        n_lab = np.argmax(label_l[np.argsort(norms,axis=1)[:,0:10]], axis=2)
        n_wts = ( np.exp(-np.sort(norms,axis=1)[:,0:10]).T / np.sum(np.exp(-np.sort(norms,axis=1)[:,0:10]),axis=1) ).T
        contexts[remaining_samples] = CBQ.questions.context_generation(prob_mat, Y_partial[remaining_samples] ,n_lab, n_wts, costs=costs, normalize = normalize_contexts)
        print(datetime.datetime.now()-temp)
        
        # compute OFUL optimism
        logdetB = np.linalg.det(CBQ.B)
        S_hat = 1 # upper bound on mu_star 2-norm
        R = 1
        delta = 0.1
        sqrtBeta = R * np.sqrt( logdetB - CBQ.context_dim * np.log(CBQ.lambda_regul) + np.log(1/(delta**2)) ) + np.sqrt(CBQ.lambda_regul) * S_hat
    

    # REWARDS PREDICTIONS
    # WARNING : current implementation consider that contexts don't change in each batch
    if batch == 0 :
        temp_rew = (contexts.transpose(0,2,1) @ CBQ.B_inv @ contexts).diagonal(0,2)
        predicted_rewards = (contexts.transpose(0,2,1) @ CBQ.mu_cap).T.squeeze(0).T + sqrtBeta * temp_rew

    # SELECT BEST ARM      
    pred = predicted_rewards * available_arms
    idx = np.nanargmax(pred)
    rnd_samp = idx // CBQ.num_arms
    selected_arm =  idx % CBQ.num_arms
    
    #GET ANSWER, UPDATE LABEL, UPDATE AVAILABLE ARMS
    #keep trak of root questions
    if selected_arm == 0 and inactive_arm :
        num_root += 1
    
    # GET ANSWER
    if Y[rnd_samp].argmax() in CBQ.questions.tree_to_actual[selected_arm + (0 if inactive_arm else 1)]:
        answer = Y_partial[rnd_samp].numpy() * CBQ.arms_data[selected_arm]
        pos_answers +=1
    else:
        answer = (Y_partial[rnd_samp].numpy()-CBQ.arms_data[selected_arm] > 0)*1
    Y_partial[rnd_samp] = torch.from_numpy(answer)
    sample_ques[rnd_samp]['prev_ans'].append(answer)
    

    # keep track of update to training samples / duplicates due to partial labels
    # keep track of remaining questions for next round
    sample_ques[rnd_samp]['prev_ques'].append(selected_arm)
    rem_q_i, remove_q_i = rem_q(answer, CBQ.arms_data, sample_ques[rnd_samp]['prev_ques'], sample_ques[rnd_samp]['remove_ques'], root = inactive_arm)
    try:
        assert(set(rem_q_i) - set(sample_ques[rnd_samp]['remaining_ques']) == set())
    except:
        print('Assertion error', rem_q_i, sample_ques[rnd_samp]['remaining_ques'], answer, Y_partial[rnd_samp], sample_ques[rnd_samp]['prev_ques'], sample_ques[rnd_samp]['prev_ans'])

    sample_ques[rnd_samp]['remaining_ques'] = rem_q_i
    sample_ques[rnd_samp]['remove_ques'] += remove_q_i
    
    if not all_arms_available :
        available_arms[rnd_samp] = np.array([1. if (k in rem_q_i) else np.nan for k in range(CBQ.questions.num_questions)])

    all_arms.append(selected_arm)
    all_contexts.append(contexts[rnd_samp,:,selected_arm])    
    updated_points.append(rnd_samp)
    

    

    # keep track of labeling state
    if torch.sum(Y_partial[rnd_samp]) == 1:  # check if fully labeled
        sample_label_state['FL'].append(rnd_samp)  # add to fl list
        if rnd_samp in sample_label_state['PL']:  # if in PL remove from there
            sample_label_state['PL'].remove(rnd_samp)
        elif rnd_samp in sample_label_state['UL']:
            sample_label_state['UL'].remove(rnd_samp)  # if in UL remove from there
    elif torch.sum(Y_partial[rnd_samp]) != 10:  # if not FL and not UL
        if rnd_samp not in sample_label_state['PL']:  #if not in PL add there are remove from UL
            sample_label_state['PL'].append(rnd_samp)
            sample_label_state['UL'].remove(rnd_samp)

    batch += 1
    
    if batch == num_pts :

        # some traks
        root_selection.append(num_root)
        pos_answer_cnt.append(pos_answers)

        for k in range(1,11):
            Y_partial_state[k].append((np.sum(Y_partial.numpy(),axis=1)==k).sum())
        
        # TRAIN NETWORK WITH NEW SAMPLES
        partial_train_idx = list(set(np.arange(num_unlabeled)) - set(sample_label_state['UL']))
        X_train_al = torch.cat([X_l, X[partial_train_idx]])
        Y_train_al = torch.cat([Y_l, Y_partial[partial_train_idx]])
        
        net.apply(weight_reset)
        print( 'retraining number', (r+1)//num_pts  )
        train_partial(X_train_al, Y_train_al, net, optimizer, criterion, num_epochs, batch_size, device,
                        stopping = stopping, X_val = X_val, Y_val = Y_val)
    
        
        # GET REWARD
        new_loss, new_acc = test_partial(X_val, Y_val, net, criterion, batch_size,  device)
        reward_acc = new_acc - init_acc
        init_acc = new_acc
        reward_loss = init_loss - new_loss 
        init_loss = new_loss
        
        # get global annotation cost
        anno_cost_local = CBQ.questions.cost_annotation(costs=costs)[all_arms]
        anno_cost = anno_cost_local.sum()
        num_full = sum([1 for arm in all_arms if arm in CBQ.questions.leaves])
        num_partial = num_pts - num_root -num_full

        partial_qtype_cnt.append(num_partial)
        full_qtype_cnt.append(num_full)
        anno_cnt.append(anno_cost)

        
       
        
        # Get prob vec/emb vec from network for context
        prob_mat_update, emb_mat_update = get_prob_and_emb(inputs, net, 100,  device, dim = net.embSize)


        # UPDATE BANDIT PARAMS
        for i, samp in enumerate(updated_points) :
            sample_idx = input_index[samp]
            aig = (entropy_calc(prob_mat[sample_idx].unsqueeze(0)) - entropy_calc(prob_mat_update[sample_idx].unsqueeze(0))).item()
            cosine = prob_mat[sample_idx].dot( prob_mat_update[sample_idx] ) / ( np.linalg.norm( prob_mat[sample_idx] ) * np.linalg.norm( prob_mat_update[sample_idx] ) )
            cosine = (2/np.pi) * np.arccos( cosine.clamp(min = -1,max = 1) )
            
            reward_vect = np.array([reward_acc,
                                    reward_loss,
                                    aig,
                                    cosine,
                                    anno_cost_local[i]
                                    ])
            lambdas = np.array([lambda_acc,
                                lambda_loss,
                                lambda_aig,
                                lambda_cos,
                                lambda_ann ])
            normalization = np.array([1,
                                      1,
                                      np.log(num_classes),
                                      1,
                                      np.max(costs) ])
            if normalize_reward :
                reward_vect = reward_vect / normalization
            
            if reward_type == 'global':
                assert(lambda_aig == 0 and lambda_cos == 0)
                reward_vect[-1] = anno_cost
            
            reward = np.sum(reward_vect * lambdas).item() + bias
            
            reward_real.append(reward)
            CBQ.bc = all_contexts[i]
            CBQ.update_tsp_tmp([all_arms[i]], reward, i)

        CBQ.update_thompson_sampling()
        
        mu_array = np.concatenate((mu_array, CBQ.mu_cap),axis=1)
    
        loss , acc = test_partial(X_test, Y_test, net, criterion, batch_size, device)
        test_acc.append(acc)
        test_loss.append(loss)
        print('acc :', acc)
            
        batch = 0


    cnt = 0
    for smp in sample_ques.keys():
        cnt += len(sample_ques[smp]['prev_ques'])
    ques_cnt.append(cnt)
    if not fully_label_same_set:
        if torch.sum(Y_partial) == num_unlabeled:
            print('num_rounds:', r+1, CBQ.mu_cap)
            break
    else:
        if torch.sum(Y_partial) == num_classes*(num_unlabeled-num_pts) + num_pts:
            print('num_rounds:', r+1, CBQ.mu_cap)
            break
        
# TRAIN W/ COMPLETE LABELS BASELINE
X_train_al = torch.cat([X_l, X])
Y_train_al = torch.cat([Y_l, Y])

net.apply(weight_reset)
print('last training')
train_partial(X_train_al, Y_train_al, net, optimizer, criterion, num_epochs, batch_size, device,
                stopping = stopping, X_val = X_val, Y_val = Y_val)
loss_full , acc_full = test_partial(X_test, Y_test, net, criterion, batch_size,  device)

print('acc with complete labels:', acc_full)
print('algebric distance to fully labeled set : \n',
      'delta loss :', loss_full - test_loss[-1], '\n',
      'delta accuracy :', acc_full - test_acc[-1] )



print(datetime.datetime.now()-now)
