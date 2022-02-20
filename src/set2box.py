import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import numpy as np
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange

import utils

EPS = 1e-10
MAX = 1e6
MIN = -1e6
    
class model(nn.Module):
    def __init__(self, num_item, hidden_dim, beta, attention='scp'):
        super(model, self).__init__()
        self.beta = beta
        self.dim = hidden_dim
        self.num_item = num_item
        
        if attention == 'scp':
            self.pool = self.attention_scp
        
        self.center_attention = nn.Parameter(torch.empty(self.dim))
        self.radius_attention = nn.Parameter(torch.empty(self.dim))
        
        self.center_embedding = nn.Embedding(self.num_item, self.dim)
        self.radius_embedding = nn.Embedding(self.num_item, self.dim)
        
        self.clip_max = torch.FloatTensor([1.0])
        self.init_weights()
        self.radius_embedding.weight.data = self.radius_embedding.weight.data.clamp(min=EPS)
        self.loss = nn.MSELoss(reduction='sum')
        
    def init_weights(self):
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.radius_embedding.weight, mean=0.0, std=0.1)
        self.center_embedding.weight.data.div_(torch.max(torch.norm(self.center_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(self.center_embedding.weight.data))
        self.radius_embedding.weight.data.div_(torch.max(torch.norm(self.radius_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(self.radius_embedding.weight.data))
        
        stdv = 1. / math.sqrt(self.dim)
        nn.init.uniform_(self.center_attention, -stdv, stdv)
        nn.init.uniform_(self.radius_attention, -stdv, stdv)
    
    def embed_set(self, S, M, instances=None):
        if instances is not None:
            batch_sets, _instances = torch.unique(instances, return_inverse=True)
            S_batch = S[batch_sets]
            M_batch = M[batch_sets]
            emb_center = self.pool(S_batch, M_batch, self.center_embedding.weight, self.center_attention, False)
            emb_radius = self.pool(S_batch, M_batch, self.radius_embedding.weight, self.radius_attention, True)
            emb_center = emb_center[_instances]
            emb_radius = emb_radius[_instances]
        else:
            emb_center = self.pool(S, M, self.center_embedding.weight, self.center_attention, False)
            emb_radius = self.pool(S, M, self.radius_embedding.weight, self.radius_attention, True)
        return emb_center, emb_radius  
    
    def forward(self, S, M, instances, overlaps):
        emb_center, emb_radius = self.embed_set(S, M, instances)
        
        c_i, c_j, c_k = emb_center[:,0,:], emb_center[:,1,:], emb_center[:,2,:]
        r_i, r_j, r_k = emb_radius[:,0,:], emb_radius[:,1,:], emb_radius[:,2,:]
        
        m_i, m_j, m_k = c_i - r_i, c_j - r_j, c_k - r_k
        M_i, M_j, M_k = c_i + r_i, c_j + r_j, c_k + r_k
        
        C_i = torch.sum(torch.log(F.softplus(M_i - m_i, self.beta) + EPS), 1)
        C_j = torch.sum(torch.log(F.softplus(M_j - m_j, self.beta) + EPS), 1)
        C_k = torch.sum(torch.log(F.softplus(M_k - m_k, self.beta) + EPS), 1)
        C_ij = torch.sum(torch.log(F.softplus(torch.min(M_i, M_j) - torch.max(m_i, m_j), self.beta) + EPS), 1)
        C_jk = torch.sum(torch.log(F.softplus(torch.min(M_j, M_k) - torch.max(m_j, m_k), self.beta) + EPS), 1)
        C_ki = torch.sum(torch.log(F.softplus(torch.min(M_k, M_i) - torch.max(m_k, m_i), self.beta) + EPS), 1)
        C_ijk = torch.sum(torch.log(F.softplus(torch.min(M_i, torch.min(M_j, M_k)) - torch.max(m_i, torch.max(m_j, m_k)), self.beta) + EPS), 1)

        Z = torch.max(torch.max(C_i), torch.max(torch.max(C_j), torch.max(C_k)))
        
        C_i = torch.exp(C_i - Z)
        C_j = torch.exp(C_j - Z)
        C_k = torch.exp(C_k - Z)
        C_ij = torch.exp(C_ij - Z)
        C_jk = torch.exp(C_jk - Z)
        C_ki = torch.exp(C_ki - Z)
        C_ijk = torch.exp(C_ijk - Z)
        S = C_i + C_j + C_k + C_ij + C_jk + C_ki + C_ijk
        
        pred = torch.stack((C_i/S, C_j/S, C_k/S, C_ij/S, C_jk/S, C_ki/S, C_ijk/S), 1)
        loss = self.loss(pred, overlaps)
        return loss
    
    def attention_scp(self, S, M, X, A, size_reg=False):
        edges = torch.nonzero(M).t()
        edges[1,:] = S[edges[0], edges[1]]
        
        att = torch.matmul(X, A)
        weight = torch_scatter.scatter_softmax(att[edges[1]], edges[0])
        a = torch_scatter.scatter_sum(X[edges[1]] * weight.unsqueeze(1), edges[0], dim=0)
        
        att2 = torch.sum(X[edges[1]] * a[edges[0]], 1)
        weight2 = torch_scatter.scatter_softmax(att2, edges[0])
        emb = torch_scatter.scatter_sum(X[edges[1]] * weight2.unsqueeze(1), edges[0], dim=0)
        
        if size_reg:
            sizes = torch.sum(M, 1).repeat_interleave(self.dim).view(len(emb), -1)
            emb = emb * (sizes ** (1.0 / self.dim))
        
        return emb
