# Active-Question-Thompson-Sampling

## Goal

This public repository shares, for reproducibility and experiments, the code used in the paper :
"ACTIVE QUESTION LEARNING: LEARNING A PARTIAL ANNOTATION POLICY IN HIERARCHICAL LABEL SPACES"

We propose the framework Active Question Learning (AQL), designed for Active Learning with Partial Labels.
We also propose the code of the algorithms AQTS-D, AQTS-CR and AQTS-R.
Finally, we propose several hierarchies on CIFAR-10, CIFAR-100, and Tiny ImageNet

## Contributors
Anonymous authors

## Setup
For now, please feel free to use the ```env.yml``` for setting up conda environment and refer to it for any package versions - it contains several extra packages.

## Key elements
Usefull code is in python_code directory.
- ```train_AQTS.py```: contains the run of main learning algorithm, selecting iteratively samples and questions. DEPRECATED
- ```ALPF.py```: algorithm ALPF from Hu et al. - Active Learning with Partial Feedback, 2018 - http://arxiv.org/abs/1802.07427
-  ```classic_AL.py```: Classic active learning baseline, sample selection availables of least confident sampling, entropy sampling and random sampling
- ```networks.py```: network definitions and related operations
- ```datasets.py```: code for dataset imports
- ```questions.py```: contains a class computing hierarchies, questions and related contextual vectors
- ```al_utils.py```: contains a class for contextual bandit parameters update
- ```config.py```: contains all the data / network / bandit setup info


