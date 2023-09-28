# AQ--internship

## Project Active Question

Our goal is to implement and study an active learning algorithm to partially labelize datasets at the lowest cost.
We assume all samples in a dataset don't need complete annotation. We use thompson sampling and hierearchical structure on labels to decide which points need complete information and which ones need partial ones.

## Contributors
This work is originally designed by Sanjeel Pareck, previously post-doc at Telecom Paris. The following work is updated by Ignacio Laurenty, previously Master 2 intern at Telecom Paris. 
The work is supervised by Florence d'Alché-Buc and Ekhiñe Irurozki.   

## Setup
For now, please feel free to use the ```env.yml``` for setting up conda environment and refer to it for any package versions - it contains several extra packages.

## Key elements
Usefull code is in python_code directory.
- ```train_AQTS.py```: contains the run of main learning algorithm, selecting iteratively samples and questions. DEPRECATED
- ```train_AQTS_paired.py```: run of AQTS version selecting pairs of exemple and question. DEPRECATED
- ```train_AQTS_v3.py```: the main version to use, implemented with Thompson Sampling.
- ```train_OFUL.py``` : Second main version, implemented with OFUL algorithm.
- ```ALPF.py```: algorithm ALPF from Hu2018 - http://arxiv.org/abs/1802.07427
-  ```classic_AL.py```: Classic active learning baseline, sample selection availables of least confident sampling, entropy sampling and random sampling
- ```networks.py```: network definitions and related operations
- ```resnets.py``` : implementation of resnet Networks adapted to embedding extraction
- ```datasets.py```: code for dataset imports
- ```questions.py```: contains a class computing hierarchies, questions and related contextual vectors
- ```al_utils.py```: contains a class for contextual bandit parameters update
- ```config.py```: contains all the data / network / bandit setup info


### Additional files
- ```pmath.py```: contains hyperbolic space math implementation
- ```extras``` directory: contains hyperbolic embedding of dataset label spaces computed using combinatorial construction from https://github.com/HazyResearch/hyperbolics

### Additional directory
- ```python_code/version_1```: contains first version of AQTS without the inactive arm

### Contact
Feel free to contact me at ignacio.laurenty@telecom-paris.fr for comments or informations.

