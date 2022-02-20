import os
import time
import argparse
import warnings
import numpy as np
import pickle as pkl
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import set2box
import set2boxp

EPS = 1e-10

class Evaluation():
    def __init__(self, sets):
        super().__init__()
        self.sets = sets
        self.idx = {}
        self.ans = {}
        self.ground_truth()
        
    def ground_truth(self, num_samples=100000, rand_seed=2022):
        np.random.seed(rand_seed)
        
        for dtype in ['train', 'valid', 'test']:
            self.idx[dtype], self.ans[dtype] = None, {}
    
            self.idx[dtype] = {
                'i': torch.from_numpy(np.random.randint(len(self.sets[dtype]), size=num_samples)),
                'j': torch.from_numpy(np.random.randint(len(self.sets[dtype]), size=num_samples))
            }
            
            ### Ground-truth similarites
            for metric in ['ji', 'di', 'oc', 'cs']:
                self.ans[dtype][metric] = utils.gt_pairwise_similarity(self.sets[dtype], self.idx[dtype]['i'], self.idx[dtype]['j'], metric)

    def pairwise_similarity(self, our_model, S, M, beta, dtype, set2box=True):
        our_model.eval()
        device = next(our_model.parameters()).device
        
        num_sets = len(S)
        idx = torch.tensor(np.arange(num_sets))
        idx_i = self.idx[dtype]['i']
        idx_j = self.idx[dtype]['j']

        c, r = [], []
        batches = utils.generate_batches(num_sets, 4000, shuffle=False)
        for batch in batches:
            S_temp = S[batch].to(device)
            M_temp = M[batch].to(device)
            if set2box:
                _c, _r = our_model.embed_set(S_temp, M_temp)
            else:
                _, _, _c, _r = our_model.embed_set(S_temp, M_temp)
            c.append(_c.cpu().detach())
            r.append(_r.cpu().detach())
            del S_temp, M_temp
        c = torch.cat(c, 0)
        r = torch.cat(r, 0)
        c, r = c.cpu(), r.cpu()

        m_i, m_j = c[idx] - r[idx], c[idx] - r[idx]
        M_i, M_j = c[idx] + r[idx], c[idx] + r[idx]

        m_i, M_i = m_i[idx_i], M_i[idx_i]
        m_j, M_j = m_j[idx_j], M_j[idx_j]
        
        inter_ij = torch.sum(torch.log(F.softplus(torch.min(M_i, M_j) - torch.max(m_i, m_j), beta) + EPS), 1)
        union_ij = torch.sum(torch.log(F.softplus(torch.max(M_i, M_j) - torch.min(m_i, m_j), beta) + EPS), 1)
        V_i = torch.sum(torch.log(F.softplus(M_i - m_i, beta) + EPS), 1)
        V_j = torch.sum(torch.log(F.softplus(M_j - m_j, beta) + EPS), 1)
        
        Z = torch.max(union_ij)
        
        inter_ij = torch.exp(inter_ij - Z)
        union_ij = torch.exp(union_ij - Z)
        V_i = torch.exp(V_i - Z)
        V_j = torch.exp(V_j - Z)
        
        pred = {
            'ji': (inter_ij / union_ij).detach(),
            'oc': (inter_ij / torch.min(V_i, V_j)).detach(),
            'di': ((2 * inter_ij) / (V_i + V_j)).detach(),
            'cs': (inter_ij / ((V_i * V_j)**0.5)).detach()
        }

        return pred
