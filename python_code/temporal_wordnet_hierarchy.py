import numpy as np
import pickle

# CIFAR 10: all off-the-shelf questions (not presented in the paper)
def cifar10_questions():
    # Q = np.diag(np.ones(10, dtype=np.bool))
    Q = np.diag(np.ones(10, dtype=np.int))
    return Q

# CIFAR 10: all possible questions (not presented in the paper)
def cifar10_all_questions():
    K = 10
    # Q = np.zeros((2**K, K), dtype=np.bool)
    Q = np.zeros((2**K, K), dtype=np.int)
    for i in range(0, 2**K):
        num = i
        for j in range(K)[::-1]:
            if num >= 2**j:
                Q[i][j] = 1
                num -= 2**j
    # it is dumb to ask a question that involves no none or all choices
    Q = Q[1:-1]
    return Q

# CIFAR 10: all questions derived from wordnet (presented in the paper)
def cifar10_questions_wordnet():
    K = 10
    # Q = np.diag(np.ones(K, dtype=np.bool))
    Q = np.diag(np.ones(K, dtype=np.int))

    with open('data/cifar10/cifar-10-batches-py/batches.meta', 'rb') as f:
        meta = pickle.load(f)

    labels = {}
    clubs = {}
    from nltk.corpus import wordnet
    for label_index, label_name in enumerate(meta['label_names']):
        labels[label_name] = label_index
        synsets = wordnet.synsets(label_name, pos='n')
        for synset in synsets:
            paths = synset.hypernym_paths()
            for path in paths:
                for hypernym in path:
                    hypernym_name = hypernym.name()
                    if hypernym_name not in clubs:
                        clubs[hypernym_name] = set()
                    clubs[hypernym_name].add(label_name)

    # Q2 = np.zeros((len(clubs), K), dtype=np.bool)
    Q2 = np.zeros((len(clubs), K), dtype=np.int)
    hypernym_names = []
    for i, (hypernym_name, label_names) in enumerate(clubs.items()):
        hypernym_names.append(hypernym_name)
        for label_name in label_names:
            label_index = labels[label_name]
            Q2[i][label_index] = True

    # prune Q2
    I = np.where(np.logical_and(Q2.sum(axis=-1)>1, Q2.sum(axis=-1)<K))
    Q2 = Q2[I[0]]
    hypernym_names = [hypernym_names[i] for i in I[0]]

    Q = np.concatenate((Q, Q2))
    clubs = meta['label_names'] + hypernym_names

    # NOTE: we do sometimes get weird clubs like (bird, cat, dog, frog)
    # and their common ancestor is person
    # this is because they can all be used to describe a person (oh well)

    return Q

# CIFAR 100: all off-the-shelf questions (not presented in the paper)
#   these questions are derived from CIFAR100's two layer label tree
def cifar100_questions():
    f = open('data/cifar100/cifar-100-python/train', 'rb')
    data = pickle.load(f)
    coarse_labels = data['coarse_labels']
    fine_labels = data['fine_labels']
    assert(len(coarse_labels) == len(fine_labels))

    K1 = len(np.unique(coarse_labels))
    K2 = len(np.unique(fine_labels))

    # Q = np.zeros((K1+K2, K2), dtype=np.bool)
    Q = np.zeros((K1+K2, K2), dtype=np.int)
    for i in range(len(coarse_labels)):
        cl, fl = coarse_labels[i], fine_labels[i]
        Q[fl][fl] = True
        Q[K2+cl][fl] = True
    return Q

# CIFAR 100: all questions derived from wordnet (presented in the paper)
def cifar100_questions_wordnet():
    with open('../../external/data/cifar100/cifar-100-python/train', 'rb') as f:
        data = pickle.load(f)

    coarse_labels = data['coarse_labels']
    fine_labels = data['fine_labels']

    assert(len(coarse_labels) == len(fine_labels))

    K1 = len(np.unique(coarse_labels))
    K2 = len(np.unique(fine_labels))

    # Q = np.zeros((K1+K2, K2), dtype=np.bool)
    Q = np.zeros((K1+K2, K2), dtype=np.int)
    for i in range(len(coarse_labels)):
        cl, fl = coarse_labels[i], fine_labels[i]
        Q[fl][fl] = True
        Q[K2+cl][fl] = True

    # add questions constructed based on wordnet
    with open('../../external/data/cifar100/cifar-100-python/meta', 'rb') as f:
        meta = pickle.load(f)

    labels = {}
    clubs = {}
    from nltk.corpus import wordnet
    for fine_label_index, fine_label_name in enumerate(meta['fine_label_names']):
        labels[fine_label_name] = fine_label_index
        synsets = wordnet.synsets(fine_label_name, pos='n')
        for synset in synsets:
            paths = synset.hypernym_paths()
            for path in paths:
                for hypernym in path:
                    hypernym_name = hypernym.name()
                    if hypernym_name not in clubs:
                        clubs[hypernym_name] = set()
                    clubs[hypernym_name].add(fine_label_name)

    # Q2 = np.zeros((len(clubs), K2), dtype=np.bool)
    Q2 = np.zeros((len(clubs), K2), dtype=np.int)
    hypernym_names = []
    for i, (hypernym_name, fine_label_names) in enumerate(clubs.items()):
        hypernym_names.append(hypernym_name)
        for fine_label_name in fine_label_names:
            label_index = labels[fine_label_name]
            Q2[i][label_index] = True

    # prune Q2
    I = np.where(np.logical_and(Q2.sum(axis=-1)>1, Q2.sum(axis=-1)<K2))
    Q2 = Q2[I[0]]
    hypernym_names = [hypernym_names[i] for i in I[0]]

    Q = np.concatenate((Q, Q2))
    clubs = meta['fine_label_names'] + meta['coarse_label_names'] + hypernym_names

    assert(Q.shape[0] == len(clubs))

    return Q

# Tiny ImageNet: all questions derived from wordnet (presented in the paper)
def tinyimagenet200_questions_wordnet(return_clubs=False):
    K = 200
    # Q = np.diag(np.ones(K, dtype=np.bool))
    Q = np.diag(np.ones(K, dtype=np.int))

    from nltk.corpus import wordnet
    with open('data/tinyimagenet200/wnids.txt') as f:
        wnids = [l.split()[0] for l in f]
    synsets = [wordnet.synset_from_pos_and_offset(wnid[0], int(wnid[1:])) for wnid in wnids]
    meta = {'label_names': [synset.name() for synset in synsets]}

    labels = {synset.name(): i for i, synset in enumerate(synsets)}
    clubs = {}
    for synset in synsets:
        paths = synset.hypernym_paths()
        for path in paths:
            for hypernym in path:
                hypernym_name = hypernym.name()
                if hypernym_name not in clubs:
                    clubs[hypernym_name] = set()
                clubs[hypernym_name].add(synset.name())

    # Q2 = np.zeros((len(clubs), K), dtype=np.bool)
    Q2 = np.zeros((len(clubs), K), dtype=np.int)
    hypernym_names = []
    for i, (hypernym_name, fine_label_names) in enumerate(clubs.items()):
        hypernym_names.append(hypernym_name)
        for fine_label_name in fine_label_names:
            label_index = labels[fine_label_name]
            Q2[i][label_index] = True

    # prune Q2
    I = np.where(np.logical_and(Q2.sum(axis=-1)>1, Q2.sum(axis=-1)<K))
    Q2 = Q2[I[0]]
    hypernym_names = [hypernym_names[i] for i in I[0]]

    Q = np.concatenate((Q, Q2))
    clubs = meta['label_names'] + hypernym_names

    assert(Q.shape[0] == len(clubs))

    if not return_clubs:
        return Q
    else:
        return Q, clubs
    
#%%
# -*- coding: utf-8 -*-
# https://github.com/07Agarg/HIERMATCH/blob/main/HIERMATCH-cifar-100-partial/cifar100_get_tree_target_3.py
# @author: Ashima


import numpy as np
import torch
import torch.nn as nn

trees = [
	[0, 4, 5],
    [1, 1, 6],
    [2, 14, 3],
    [3, 8, 3],
    [4, 0, 6],
    [5, 6, 0],
    [6, 7, 4],
    [7, 7, 4],
    [8, 18, 7],
    [9, 3, 0],
    [10, 3, 0],
    [11, 14, 3],
    [12, 9, 1],
    [13, 18, 7],
    [14, 7, 4],
    [15, 11, 3],
    [16, 3, 0],
    [17, 9, 1],
    [18, 7, 4],
    [19, 11, 3],
    [20, 6, 0],
    [21, 11, 3],
    [22, 5, 0],
    [23, 10, 2],
    [24, 7, 4],
    [25, 6, 0],
    [26, 13, 4],
    [27, 15, 4],
    [28, 3, 0],
    [29, 15, 4],
    [30, 0, 6],
    [31, 11, 3],
    [32, 1, 6],
    [33, 10, 2],
    [34, 12, 3],
    [35, 14, 3],
    [36, 16, 3],
    [37, 9, 1],
    [38, 11, 3],
    [39, 5, 0],
    [40, 5, 0],
    [41, 19, 7],
    [42, 8, 3],
    [43, 8, 3],
    [44, 15, 4],
    [45, 13, 4],
    [46, 14, 3],
    [47, 17, 5],
    [48, 18, 7],
    [49, 10, 2],
    [50, 16, 3],
    [51, 4, 5],
    [52, 17, 5],
    [53, 4, 5],
    [54, 2, 5],
    [55, 0, 6],
    [56, 17, 5],
    [57, 4, 5],
    [58, 18, 7],
    [59, 17, 5],
    [60, 10, 2],
    [61, 3, 0],
    [62, 2, 5],
    [63, 12, 3],
    [64, 12, 3],
    [65, 16, 3],
    [66, 12, 3],
    [67, 1, 6],
    [68, 9, 1],
    [69, 19, 7],
    [70, 2, 5],
    [71, 10, 2],
    [72, 0, 6],
    [73, 1, 6],
    [74, 16, 3],
    [75, 12, 3],
    [76, 9, 1],
    [77, 13, 4],
    [78, 15, 4],
    [79, 13, 4],
    [80, 16, 3],
    [81, 19, 7],
    [82, 2, 5],
    [83, 4, 5],
    [84, 6, 0],
    [85, 19, 7],
    [86, 5, 0],
    [87, 5, 0],
    [88, 8, 3],
    [89, 19, 7],
    [90, 18, 7],
    [91, 1, 6],
    [92, 2, 5],
    [93, 15, 4],
    [94, 6, 0],
    [95, 0, 6],
    [96, 17, 5],
    [97, 8, 3],
    [98, 14, 3],
    [99, 13, 4]]

def get_order_family_target_(target, i):
    if target == -1:
        return target
    return trees[target][i]

def get_order_target(targets, level):
    order_target_list = []
    for i in range(targets.size(0)):
        target = torch.argmax(targets[i])
        order_target_list.append(trees[target][level+1])
    
    return np.array(order_target_list)

def get_order_family_target(targets):
    order_target_list = []
    family_target_list = []
    for i in range(targets.size(0)):
        order_target_list.append(trees[targets[i]][1])
        family_target_list.append(trees[targets[i]][2])

    order_target_list = torch.from_numpy(np.array(order_target_list))
    family_target_list = torch.from_numpy(np.array(family_target_list))

    return order_target_list, family_target_list
    