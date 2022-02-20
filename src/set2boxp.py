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
    def __init__(self, num_item, hidden_dim, beta, K, D, tau, attention='scp'):
        super(model, self).__init__()
        self.beta = beta
        self.dim = hidden_dim
        self.K = K
        self.D = D
        self.tau = tau
        self.num_item = num_item
        
        if attention == 'scp':
            self.pool = self.attention_scp
        
        self.center_attention = nn.Parameter(torch.empty(self.dim))
        self.radius_attention = nn.Parameter(torch.empty(self.dim))
        
        self.center_embedding = nn.Embedding(self.num_item, self.dim)
        self.radius_embedding = nn.Embedding(self.num_item, self.dim)
        
        self.center_centroid = nn.Embedding(self.K, self.dim)
        self.radius_centroid = nn.Embedding(self.K, self.dim)
        
        self.clip_max = torch.FloatTensor([1.0])
        self.init_weights()
        self.loss = nn.MSELoss(reduction='sum')
        
    def init_weights(self):
        nn.init.normal_(self.center_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.radius_embedding.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.center_centroid.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.radius_centroid.weight, mean=0.0, std=0.1)
        self.center_embedding.weight.data.div_(torch.max(torch.norm(self.center_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(self.center_embedding.weight.data))
        self.radius_embedding.weight.data.div_(torch.max(torch.norm(self.radius_embedding.weight.data, 2, 1, True), self.clip_max).expand_as(self.radius_embedding.weight.data))
        self.center_centroid.weight.data.div_(torch.max(torch.norm(self.center_centroid.weight.data, 2, 1, True), self.clip_max).expand_as(self.center_centroid.weight.data))
        self.radius_centroid.weight.data.div_(torch.max(torch.norm(self.radius_centroid.weight.data, 2, 1, True), self.clip_max).expand_as(self.radius_centroid.weight.data))
        
        self.radius_embedding.weight.data = self.radius_embedding.weight.data.clamp(min=EPS)
        self.radius_centroid.weight.data = self.radius_centroid.weight.data.clamp(min=EPS)
        
        stdv = 1. / math.sqrt(self.dim)
        nn.init.uniform_(self.center_attention, -stdv, stdv)
        nn.init.uniform_(self.radius_attention, -stdv, stdv)
    
    def diff_softmax(self, x, tau, dim):
        y_soft = (x / tau).softmax(dim)
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(x, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
        return ret
    
    def quantize(self, center, radius):
        num_queries = len(center)
        
        m_queries = center - radius
        M_queries = center + radius
        m_centroids = self.center_centroid.weight - self.radius_centroid.weight
        M_centroids = self.center_centroid.weight + self.radius_centroid.weight
        
        V_queries = torch.log(F.softplus(M_queries - m_queries, self.beta) + EPS)
        V_centroids = torch.log(F.softplus(M_centroids - m_centroids, self.beta) + EPS)
        V_inter = torch.log(F.softplus(torch.min(M_queries.unsqueeze(1), M_centroids.unsqueeze(0)) - torch.max(m_queries.unsqueeze(1), m_centroids.unsqueeze(0)), self.beta) + EPS)
        
        V_queries = torch.sum(V_queries.view(num_queries, self.D, -1), dim=-1)
        V_centroids = torch.sum(V_centroids.view(self.K, self.D, -1), dim=-1)
        V_inter = torch.sum(V_inter.view(num_queries, self.K, self.D, -1), dim=-1)
        
        Z = torch.max(torch.max(V_queries), torch.max(V_centroids))
        
        V_queries = torch.exp(V_queries - Z)
        V_centroids = torch.exp(V_centroids - Z)
        V_inter = torch.exp(V_inter - Z)
        
        # Overlap(A, B) = 0.5 * ((A ^ B) / A + (A ^ B) / B)
        overlap = (V_inter * (V_queries.unsqueeze(1) + V_centroids.unsqueeze(0))) / (V_queries.unsqueeze(1) * V_centroids.unsqueeze(0)) / 2.0
        weight = self.diff_softmax(overlap, tau=self.tau, dim=1)
        
        rec_emb_center = weight.unsqueeze(-1) * self.center_centroid.weight.view(1, self.K, self.D, -1)
        rec_emb_radius = weight.unsqueeze(-1) * self.radius_centroid.weight.view(1, self.K, self.D, -1)
        
        rec_emb_center = torch.sum(rec_emb_center, 1).view(num_queries, self.dim)
        rec_emb_radius = torch.sum(rec_emb_radius, 1).view(num_queries, self.dim)
        return rec_emb_center, rec_emb_radius
    
    def embed_set(self, S, M, instances=None):
        if instances is not None:
            batch_sets, _instances = torch.unique(instances, return_inverse=True)
            S_batch = S[batch_sets]
            M_batch = M[batch_sets]
            emb_center = self.pool(S_batch, M_batch, self.center_embedding.weight, self.center_attention, False)
            emb_radius = self.pool(S_batch, M_batch, self.radius_embedding.weight, self.radius_attention, True)
            emb_center_q, emb_radius_q = self.quantize(emb_center, emb_radius)
            emb_center = emb_center[_instances]
            emb_radius = emb_radius[_instances]
            emb_center_q = emb_center_q[_instances]
            emb_radius_q = emb_radius_q[_instances]
        else:
            emb_center = self.pool(S, M, self.center_embedding.weight, self.center_attention, False)
            emb_radius = self.pool(S, M, self.radius_embedding.weight, self.radius_attention, True)
            emb_center_q, emb_radius_q = self.quantize(emb_center, emb_radius)
        return emb_center, emb_radius, emb_center_q, emb_radius_q
    
    def forward(self, S, M, instances, overlaps, joint=True):
        emb_center, emb_radius, emb_center_q, emb_radius_q = self.embed_set(S, M, instances)
        
        c_i, c_j, c_k = emb_center[:,0,:], emb_center[:,1,:], emb_center[:,2,:]
        r_i, r_j, r_k = emb_radius[:,0,:], emb_radius[:,1,:], emb_radius[:,2,:]
        c_q_i, c_q_j, c_q_k = emb_center_q[:,0,:], emb_center_q[:,1,:], emb_center_q[:,2,:]
        r_q_i, r_q_j, r_q_k = emb_radius_q[:,0,:], emb_radius_q[:,1,:], emb_radius_q[:,2,:]
        
        m_i, m_j, m_k = c_i - r_i, c_j - r_j, c_k - r_k
        M_i, M_j, M_k = c_i + r_i, c_j + r_j, c_k + r_k
        m_q_i, m_q_j, m_q_k = c_q_i - r_q_i, c_q_j - r_q_j, c_q_k - r_q_k
        M_q_i, M_q_j, M_q_k = c_q_i + r_q_i, c_q_j + r_q_j, c_q_k + r_q_k
        
        # 1-box
        C_i = torch.sum(torch.log(F.softplus(M_i - m_i, self.beta) + EPS), 1)
        C_j = torch.sum(torch.log(F.softplus(M_j - m_j, self.beta) + EPS), 1)
        C_k = torch.sum(torch.log(F.softplus(M_k - m_k, self.beta) + EPS), 1)
        C_qi = torch.sum(torch.log(F.softplus(M_q_i - m_q_i, self.beta) + EPS), 1)
        C_qj = torch.sum(torch.log(F.softplus(M_q_j - m_q_j, self.beta) + EPS), 1)
        C_qk = torch.sum(torch.log(F.softplus(M_q_k - m_q_k, self.beta) + EPS), 1)
        
        # 2-box
        C_i_j = torch.sum(torch.log(F.softplus(torch.min(M_i, M_j) - torch.max(m_i, m_j), self.beta) + EPS), 1)
        C_i_qj = torch.sum(torch.log(F.softplus(torch.min(M_i, M_q_j) - torch.max(m_i, m_q_j), self.beta) + EPS), 1)
        C_qi_j = torch.sum(torch.log(F.softplus(torch.min(M_q_i, M_j) - torch.max(m_q_i, m_j), self.beta) + EPS), 1)
        C_qi_qj = torch.sum(torch.log(F.softplus(torch.min(M_q_i, M_q_j) - torch.max(m_q_i, m_q_j), self.beta) + EPS), 1)
        C_j_k = torch.sum(torch.log(F.softplus(torch.min(M_j, M_k) - torch.max(m_j, m_k), self.beta) + EPS), 1)
        C_j_qk = torch.sum(torch.log(F.softplus(torch.min(M_j, M_q_k) - torch.max(m_j, m_q_k), self.beta) + EPS), 1)
        C_qj_k = torch.sum(torch.log(F.softplus(torch.min(M_q_j, M_k) - torch.max(m_q_j, m_k), self.beta) + EPS), 1)
        C_qj_qk = torch.sum(torch.log(F.softplus(torch.min(M_q_j, M_q_k) - torch.max(m_q_j, m_q_k), self.beta) + EPS), 1)
        C_k_i = torch.sum(torch.log(F.softplus(torch.min(M_k, M_i) - torch.max(m_k, m_i), self.beta) + EPS), 1)
        C_k_qi = torch.sum(torch.log(F.softplus(torch.min(M_k, M_q_i) - torch.max(m_k, m_q_i), self.beta) + EPS), 1)
        C_qk_i = torch.sum(torch.log(F.softplus(torch.min(M_q_k, M_i) - torch.max(m_q_k, m_i), self.beta) + EPS), 1)
        C_qk_qi = torch.sum(torch.log(F.softplus(torch.min(M_q_k, M_q_i) - torch.max(m_q_k, m_q_i), self.beta) + EPS), 1)
        
        # 3-box
        C_i_j_k = torch.sum(torch.log(F.softplus(torch.min(M_i, torch.min(M_j, M_k)) - torch.max(m_i, torch.max(m_j, m_k)), self.beta) + EPS), 1)
        C_i_j_qk = torch.sum(torch.log(F.softplus(torch.min(M_i, torch.min(M_j, M_q_k)) - torch.max(m_i, torch.max(m_j, m_q_k)), self.beta) + EPS), 1)
        C_i_qj_k = torch.sum(torch.log(F.softplus(torch.min(M_i, torch.min(M_q_j, M_k)) - torch.max(m_i, torch.max(m_q_j, m_k)), self.beta) + EPS), 1)
        C_qi_j_k = torch.sum(torch.log(F.softplus(torch.min(M_q_i, torch.min(M_j, M_k)) - torch.max(m_q_i, torch.max(m_j, m_k)), self.beta) + EPS), 1)
        C_i_qj_qk = torch.sum(torch.log(F.softplus(torch.min(M_i, torch.min(M_q_j, M_q_k)) - torch.max(m_i, torch.max(m_q_j, m_q_k)), self.beta) + EPS), 1)
        C_qi_j_qk = torch.sum(torch.log(F.softplus(torch.min(M_q_i, torch.min(M_j, M_q_k)) - torch.max(m_q_i, torch.max(m_j, m_q_k)), self.beta) + EPS), 1)
        C_qi_qj_k = torch.sum(torch.log(F.softplus(torch.min(M_q_i, torch.min(M_q_j, M_k)) - torch.max(m_q_i, torch.max(m_q_j, m_k)), self.beta) + EPS), 1)
        C_qi_qj_qk = torch.sum(torch.log(F.softplus(torch.min(M_q_i, torch.min(M_q_j, M_q_k)) - torch.max(m_q_i, torch.max(m_q_j, m_q_k)), self.beta) + EPS), 1)
        
        Z = torch.max(torch.max(C_i), torch.max(torch.max(C_j), torch.max(C_k)))
        Z_q = torch.max(torch.max(C_qi), torch.max(torch.max(C_qj), torch.max(C_qk)))
        Z = torch.max(Z, Z_q)
        
        # 1-box
        if joint:
            C_i = torch.exp(C_i - Z)
            C_j = torch.exp(C_j - Z)
            C_k = torch.exp(C_k - Z)
        C_qi = torch.exp(C_qi - Z)
        C_qj = torch.exp(C_qj - Z)
        C_qk = torch.exp(C_qk - Z)
        
        # 2-box
        if joint:
            C_i_j = torch.exp(C_i_j - Z)
            C_i_qj = torch.exp(C_i_qj - Z)
            C_qi_j = torch.exp(C_qi_j - Z)
            C_j_k = torch.exp(C_j_k - Z)
            C_j_qk = torch.exp(C_j_qk - Z)
            C_qj_k = torch.exp(C_qj_k - Z)
            C_k_i = torch.exp(C_k_i - Z)
            C_k_qi = torch.exp(C_k_qi - Z)
            C_qk_i = torch.exp(C_qk_i - Z)
        C_qi_qj = torch.exp(C_qi_qj - Z)
        C_qj_qk = torch.exp(C_qj_qk - Z)
        C_qk_qi = torch.exp(C_qk_qi - Z)
        
        # 3-box
        if joint:
            C_i_j_k = torch.exp(C_i_j_k - Z)
            C_i_j_qk = torch.exp(C_i_j_qk - Z)
            C_i_qj_k = torch.exp(C_i_qj_k - Z)
            C_qi_j_k = torch.exp(C_qi_j_k - Z)
            C_i_qj_qk = torch.exp(C_i_qj_qk - Z)
            C_qi_j_qk = torch.exp(C_qi_j_qk - Z)
            C_qi_qj_k = torch.exp(C_qi_qj_k - Z)
        C_qi_qj_qk = torch.exp(C_qi_qj_qk - Z)
        
        # Predictions
        if joint:
            S = C_i + C_j + C_k + C_i_j + C_j_k + C_k_i + C_i_j_k
            pred_1 = torch.stack((C_i/S, C_j/S, C_k/S, C_i_j/S, C_j_k/S, C_k_i/S, C_i_j_k/S), 1)

            S = C_i + C_j + C_qk + C_i_j + C_j_qk + C_qk_i + C_i_j_qk
            pred_2 = torch.stack((C_i/S, C_j/S, C_qk/S, C_i_j/S, C_j_qk/S, C_qk_i/S, C_i_j_qk/S), 1)

            S = C_i + C_qj + C_k + C_i_qj + C_qj_k + C_k_i + C_i_qj_k
            pred_3 = torch.stack((C_i/S, C_qj/S, C_k/S, C_i_qj/S, C_qj_k/S, C_k_i/S, C_i_qj_k/S), 1)

            S = C_qi + C_j + C_k + C_qi_j + C_j_k + C_k_qi + C_qi_j_k
            pred_4 = torch.stack((C_qi/S, C_j/S, C_k/S, C_qi_j/S, C_j_k/S, C_k_qi/S, C_qi_j_k/S), 1)

            S = C_i + C_qj + C_qk + C_i_qj + C_qj_qk + C_qk_i + C_i_qj_qk
            pred_5 = torch.stack((C_i/S, C_qj/S, C_qk/S, C_i_qj/S, C_qj_qk/S, C_qk_i/S, C_i_qj_qk/S), 1)

            S = C_qi + C_j + C_qk + C_qi_j + C_j_qk + C_qk_qi + C_qi_j_qk
            pred_6 = torch.stack((C_qi/S, C_j/S, C_qk/S, C_qi_j/S, C_j_qk/S, C_qk_qi/S, C_qi_j_qk/S), 1)

            S = C_qi + C_qj + C_k + C_qi_qj + C_qj_k + C_k_qi + C_qi_qj_k
            pred_7 = torch.stack((C_qi/S, C_qj/S, C_k/S, C_qi_qj/S, C_qj_k/S, C_k_qi/S, C_qi_qj_k/S), 1)
        
        S = C_qi + C_qj + C_qk + C_qi_qj + C_qj_qk + C_qk_qi + C_qi_qj_qk
        pred_8 = torch.stack((C_qi/S, C_qj/S, C_qk/S, C_qi_qj/S, C_qj_qk/S, C_qk_qi/S, C_qi_qj_qk/S), 1)
        
        if joint:
            loss_1 = self.loss(pred_1, overlaps)
            loss_2 = self.loss(pred_2, overlaps)
            loss_3 = self.loss(pred_3, overlaps)
            loss_4 = self.loss(pred_4, overlaps)
            loss_5 = self.loss(pred_5, overlaps)
            loss_6 = self.loss(pred_6, overlaps)
            loss_7 = self.loss(pred_7, overlaps)
        loss_8 = self.loss(pred_8, overlaps)
        
        if joint:
            return loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8
        return loss_8
    
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
