import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    basic = parser.add_argument_group('basic', 'basic settings')
    basic.add_argument('--seed', type=int, default=1)
    basic.add_argument('--device', type=str, default='auto')
    basic.add_argument('--optimizer', type=str, default='adam')
    basic.add_argument('--lr', type=float, default=0.001)
    basic.add_argument('--momentum', type=float, default=0.9)
    basic.add_argument('--beta1', type=float, default=0.9)
    basic.add_argument('--beta2', type=float, default=0.999)
    

    network = parser.add_argument_group('network', 'network parameters')
    network.add_argument('--architecture', type=str, default='VGG16')
    network.add_argument('--resnet_type', type =int, default=18)
    network.add_argument('--num_epochs', type=int, default=30)
    network.add_argument('--batch_size', type=int, default=512)

    initialization = parser.add_argument_group('initialization', 'initialization params')
    initialization.add_argument('--proportion', type=bool, default = False)
    initialization.add_argument('--init_lab_prop', type=float, default=0.05)
    initialization.add_argument('--init_lab_size', type=int, default = 512)
    initialization.add_argument('--initialize', type = bool, default = True)

    dataset = parser.add_argument_group('dataset', 'dataset parameters')
    dataset.add_argument('--dataset_name', type=str, default='CIFAR10')
    dataset.add_argument('--data_dir', type=str, default = '../data')
    dataset.add_argument('--num_threads', type=int, default=0)
    
    bandit = parser.add_argument_group('bandit', 'bandit generic parameters')
    bandit.add_argument('--mode', type = str, default='bandit')
    bandit.add_argument('--dag_hier', type = bool, default = True)
    bandit.add_argument('--nu2', type=float, default = 2.25)
    bandit.add_argument('--gamma_forget', type = float, default =1)
    bandit.add_argument('--lambda_regul', type= float, default = 1)
    bandit.add_argument('--S_hat', type=float, default = 1.)
    bandit.add_argument('--R', type=float, default =1.)
    bandit.add_argument('--delta',type=float, default=1.)
    bandit.add_argument('--val_prop', type=float, default=0.10)
    bandit.add_argument('--budget', type = float, default=0.5)
    
    reward = parser.add_argument_group('reward', 'reward parameters')
    reward.add_argument('--reward', type = str, default = 'local')
    reward.add_argument('--lambda_acc', type=int, default = 1)
    reward.add_argument('--lambda_loss', type=int, default = 0)
    reward.add_argument('--lambda_aig',type=int, default = 1)
    reward.add_argument('--lambda_aig_m', type=int, default = 1)
    reward.add_argument('--lambda_cos', type=int,default = 0)
    reward.add_argument('--lambda_ann', type=float, default = -0.01)
    reward.add_argument('--lambda_loc_loss', type=float, default = 0)

    arms = parser.add_argument_group('arms', 'arms parameters')
    arms.add_argument('--contexts', nargs='+', type = str, default=['eig','edc','erc', 'eig_m','ent','lc','lcq','ent_m','delta_loss','cost'])
    arms.add_argument('--inactive_arm', type=bool, default =True)
    arms.add_argument('--reject_inactive_rnd_1', type = bool, default = False)
    arms.add_argument('--inactive_zero_context', type = bool, default = True)
    arms.add_argument('--all_arms_available', type=bool, default = False)

    cost_function = parser.add_argument_group('cost function','cost function parameters')
    cost_function.add_argument('--leaf_cost', type=float, default=10.)
    cost_function.add_argument('--alpha', type = float, default = 1.)

    query = parser.add_argument_group('query', 'query parameters')
    query.add_argument('--num_pts', type = int, default = 5000  )
    query.add_argument('--pool_is_X', type=bool, default= False)
    query.add_argument('--allow_multichoice', type=bool, default = True)
    query.add_argument('--explore_all', type=bool, default = False)
    query.add_argument('--num_rounds', type = int, default = 50)
    
    statistics = parser.add_argument_group('statistics', 'statistical parameters')
    statistics.add_argument('--num_rep', type = int, default = 5)
    statistics.add_argument('--save', type =bool, default = True)
    statistics.add_argument('--save_repo', type=str, default = '../../results')
    

    args = parser.parse_args()
    return args


