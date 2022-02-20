import os
import math
import time
import random
import warnings
import numpy as np
import scipy.sparse as sp
from itertools import chain
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import utils
import enumeration
import evaluation
import set2boxp

EPS = 1e-10

warnings.filterwarnings('ignore')
args = utils.parse_args()

########## GPU Settings ##########
if torch.cuda.is_available():
    device = torch.device("cuda:" + args.gpu)
else:
    device = torch.device("cpu")
print('Device:\t', device, '\n')

########## Random Seed ##########
SEED = 2022
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

########## Read data ##########
runtime = time.time()

num_sets, sets, S, M = {}, {}, {}, {}

for dtype in ['train', 'valid', 'test']:
    num_items, num_sets[dtype], sets[dtype] = utils.read_data(args.dataset, dtype=dtype)
    S[dtype], M[dtype] = utils.incidence_matrix(sets[dtype], num_sets[dtype])
    print('{}:\t\t|V| = {}\t|E| = {}'.format(dtype, num_items, num_sets[dtype]))

memory = {
    'train': ((args.dim * 32 * 2 * args.K) + (num_sets['train'] * args.D * math.log2(args.K))) / 8000,
    'valid': ((args.dim * 32 * 2 * args.K) + (num_sets['valid'] * args.D * math.log2(args.K))) / 8000,
    'test': ((args.dim * 32 * 2 * args.K) + (num_sets['test'] * args.D * math.log2(args.K))) / 8000
}
    
print('Reading data done:\t\t{} seconds'.format(time.time() - runtime), '\n')

########## Enumerate Triplets ##########
start_time = time.time()

enumeration = enumeration.Enumeration(sets['train'])
instances, overlaps = enumeration.enumerate_instances(args.pos_instance, args.neg_instance)

instances, overlaps = instances, overlaps

print('# of instances:\t\t', len(instances))
print('Enumerating pairs done:\t\t', time.time() - start_time, '\n')

########## Prepare Evaluation ##########
start_time = time.time()

evaluation = evaluation.Evaluation(sets)

print('Preparing evaluation done:\t', time.time() - start_time, '\n')

########## Prepare Training ##########
start_time = time.time()

model = set2boxp.model(num_items, args.dim, args.beta, args.K, args.D, args.tau).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

print('Preparing training done:\t', time.time() - start_time, '\n')

########## Train Model ##########
S['train'] = S['train'].to(device)
M['train'] = M['train'].to(device)
overlaps = overlaps.to(device)

for epoch in range(1, args.epochs + 1):
    print('\nEpoch:\t', epoch)
    
    ########## Train ##########
    train_time = time.time()
    
    model.train()
    if args.lmbda == 0:
        epoch_loss = 0
    else:
        epoch_losses = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    
    batches = utils.generate_batches(len(instances), args.batch_size)
        
    for i in trange(len(batches), position=0, leave=False):
        # Forward
        batch_instances = instances[batches[i]]
        if args.lmbda == 0:
            loss_agg = model.forward(S['train'], M['train'], batch_instances, overlaps[batches[i]], False)
            epoch_loss += loss_agg.item()
        else:
            loss_1, loss_2, loss_3, loss_4, loss_5, loss_6, loss_7, loss_8 = model.forward(S['train'], M['train'], batch_instances, overlaps[batches[i]], True)
            epoch_losses[1] += loss_1.item()
            epoch_losses[2] += loss_2.item()
            epoch_losses[3] += loss_3.item()
            epoch_losses[4] += loss_4.item()
            epoch_losses[5] += loss_5.item()
            epoch_losses[6] += loss_6.item()
            epoch_losses[7] += loss_7.item()
            epoch_losses[8] += loss_8.item()
            loss_agg = args.lmbda * (loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6 + loss_7) + loss_8
        
        # Optimize
        optimizer.zero_grad()
        loss_agg.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        model.radius_embedding.weight.data = model.radius_embedding.weight.data.clamp(min=EPS)
        model.radius_centroid.weight.data = model.radius_centroid.weight.data.clamp(min=EPS)

    if args.lmbda == 0:
        print('Loss:\t', epoch_loss)
    else:
        for i in range(1, 8+1):
            print('Loss {}:\t'.format(i), epoch_losses[i])
    train_time = time.time() - train_time
    
    ########## Evaluate the Model ##########
    for dtype in ['train', 'valid', 'test']:
        pred = evaluation.pairwise_similarity(model, S[dtype], M[dtype], args.beta, dtype, False)
        for metric in ['ji', 'di', 'oc', 'cs']:
            mse = mean_squared_error(pred[metric], evaluation.ans[dtype][metric])
            print('{}_{} (MSE):\t'.format(dtype, metric) + str(mse))