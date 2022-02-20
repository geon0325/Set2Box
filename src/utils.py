import os
import copy
import math
import random
import argparse
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from itertools import chain
from tqdm import tqdm, trange
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ml1m', type=str, help='dataset')
    parser.add_argument("--gpu", default='0', type=str, help='gpu number')
    
    parser.add_argument("--batch_size", default=512, type=int, help='batch size')
    parser.add_argument("--epochs", default=200, type=int, help='number of epochs')
    parser.add_argument("--learning_rate", default=1e-3, type=float, help='learning rate')
    parser.add_argument("--dim", default=4, type=int, help='embedding dimension')
    parser.add_argument("--beta", default=1, type=float, help='beta for box smoothness')
    
    parser.add_argument("--K", default=30, type=int, help='quantization: K')
    parser.add_argument("--D", default=4, type=int, help='quantization: D')
    parser.add_argument("--tau", default=1.0, type=float, help='quantization: tau')
    parser.add_argument("--lmbda", default=0.1, type=float, help='lambda')
    
    parser.add_argument("--pos_instance", default=10, type=int, help='positive instance per set')
    parser.add_argument("--neg_instance", default=10, type=int, help='negative instance per set')
    
    return parser.parse_args()

def read_data(data, dtype='train'):
    sets = []
    with open('{}/{}/{}.txt'.format('../data', data, dtype), 'r') as f:
        num_items, num_sets = [int(x) for x in f.readline().split('\t')]
        for line in f:
            _set = [int(x) for x in line.split('\t')]
            sets.append(_set)
    assert num_sets == len(sets), 'ERROR'
    return num_items, num_sets, sets

def write_log(log_path, log_dic):
    with open(log_path, 'a') as f:
        for _key in log_dic:
            f.write(_key + '\t' + str(log_dic[_key]) + '\n')
        f.write('\n')
        
def generate_batches(data_size, batch_size, shuffle=True):
    data = np.arange(data_size)
    
    if shuffle:
        np.random.shuffle(data)
    
    batch_num = math.ceil(data_size / batch_size)
    batches = np.split(np.arange(batch_num * batch_size), batch_num)
    batches[-1] = batches[-1][:(data_size - batch_size * (batch_num - 1))]
    
    for i, batch in enumerate(batches):
        batches[i] = [data[j] for j in batch]
        
    return batches
    
def incidence_matrix(sets, num_sets):
    max_set_size = max([len(s) for s in sets])

    S = torch.ones((num_sets, max_set_size)) * -1
    M = torch.zeros((num_sets, max_set_size)).bool()
        
    for i in trange(num_sets, position=0, leave=False):
        _set = sets[i]
        S[i][:len(_set)] = torch.tensor(_set)
        M[i][:len(_set)] = torch.tensor([True] * len(_set))
    S = S.long()
    M = M.bool()
        
    return S, M

def bitcount(n):
    count = 0
    while n > 0:
        count = count + 1
        n = n & (n-1)
    return count

def gt_pairwise_similarity(sets, idx_i, idx_j, metric='ji'):
    num_sets = len(sets)
    sizes = [len(_set) for _set in sets]
    
    set2bin = []
    for _set in sets:
        s = 0
        for _item in _set:
            s = s | (1 << _item)
        set2bin.append(s)
        
    ans = []
    if metric == 'ji':
        for idx in trange(len(idx_i)):
            i, j = idx_i[idx], idx_j[idx]
            inter = bitcount(set2bin[i] & set2bin[j])
            union = bitcount(set2bin[i] | set2bin[j])
            ans.append(inter / union)
    elif metric == 'oc':
        for idx in trange(len(idx_i)):
            i, j = idx_i[idx], idx_j[idx]
            inter = bitcount(set2bin[i] & set2bin[j])
            size_i = sizes[i]
            size_j = sizes[j]
            ans.append(inter / min(size_i, size_j))
    elif metric == 'di':
        for idx in trange(len(idx_i)):
            i, j = idx_i[idx], idx_j[idx]
            inter = bitcount(set2bin[i] & set2bin[j])
            size_i = sizes[i]
            size_j = sizes[j]
            ans.append((2 * inter) / (size_i + size_j))
    elif metric == 'cs':
        for idx in trange(len(idx_i)):
            i, j = idx_i[idx], idx_j[idx]
            inter = bitcount(set2bin[i] & set2bin[j])
            size_i = sizes[i]
            size_j = sizes[j]
            ans.append(inter / ((size_i * size_j)**0.5))
    
    ans = torch.tensor(ans)    
    return ans
