# Active-Question-Thompson-Sampling

## Goal

This public repository shares, for reproducibility and experiments, the code used in the paper :
"Active Question Learning: learning a partial annotation policy in Hierarchical Label Spaces"

We propose the framework Active Question Learning (AQL), designed for Active Learning with Partial Labels.
We also propose the code of the algorithms AQTS-D, AQTS-CR and AQTS-R.
Finally, we propose several hierarchies on CIFAR-10, CIFAR-100, and Tiny ImageNet

## Contributors
Anonymous authors

## Setup
For now, please feel free to use the ```env.yml``` for setting up conda environment and refer to it for any package versions - it contains several extra packages.

## Key elements
Usefull code is in python_code directory.
- ```train_AQTS.py```: main for AQTS-R and AQTS-CR
- ```train_AQTS-D.py```: main for AQTS-D
- ```ALPF.py```: algorithm ALPF from Hu et al. - Active Learning with Partial Feedback, 2018 - http://arxiv.org/abs/1802.07427
-  ```classic_AL.py```: Classic active learning baseline, Available heuristics :  Least Confident Sampling, Entropy Sampling, Random Sampling
- ```networks.py```: network definitions and related operations
- ```datasets.py```: code for dataset imports
- ```questions.py```: contains a class computing hierarchies, questions and related contextual vectors
- ```al_utils.py```: contains a class for contextual bandit parameters update
- ```config.py```: contains all the data / network / bandit setup info


## Datasets
CIFAR datasets will be automatically downloaded when running AQTS 
Tiny ImageNet must be manually downloaded. You can download the dataset at http://cs231n.stanford.edu/tiny-imagenet-200.zip
Be aware of your data path correspond to the data_dir in config.py
