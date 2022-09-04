# Set2Box: Similarity Preserving Representation Learning for Sets

Source code for the paper [Set2Box: Similarity Preserving Representation Learning for Sets](https://github.com/geon0325/Set2Box), Geon Lee, Chanyoung Park, and Kijung Shin, [ICDM 2022](https://icdm22.cse.usf.edu/).

### Overview
Sets have been used for modeling various types of objects (e.g., a document as the set of keywords in it and a customer as the set of the items that she has purchased). Measuring similarity (e.g., Jaccard Index) between sets has been a key building block of a wide range of applications, including, plagiarism detection, recommendation, and graph compression. However, as sets have grown in numbers and sizes, the computational cost and storage required for set similarity computation have become substantial, and this has led to the development of hashing and sketching based solutions. In this work, we propose Set2Box, a learning-based approach for compressed representations of sets from which various similarity measures can be estimated accurately in constant time. The key idea is to represent sets as boxes to precisely capture overlaps of sets. Additionally, based on the proposed box quantization scheme, we design Set2Box+, which yields more concise but more accurate box representations of sets. Through extensive experiments on 8 real-world datasets, we show that, compared to baseline approaches, Set2Box+ is (a) Accurate: achieving up to 40.8× smaller estimation error while requiring 60% fewer bits to encode sets, (b) Concise: yielding up to 96.8× more concise representations with similar estimation error, and (c) Versatile: enabling the estimation of four set-similarity measures with a single set of representations.

### How to Run Set2Box
````
python main_set2box.py --dataset ml1m --gpu 0 --batch_size 512 --epochs 200 --learning_rate 1e-3 --dim 4 --beta 1 --pos_instance 10 --neg_instance 10
````

#### Arugments
````--dataset````: name of the dataset

````--gpu````: number of gpus

````--batch_size````: size of the batch

````--epochs````: number of epochs

````--learning_rate````: learning rate

````--dim````: dimension of the box

````--beta````: box smoothing paramter

````--pos_instance````: number of positive samples

````--neg_instance````: number of negative samples

### How to Run Set2Box+
````
python main_set2boxp.py --dataset ml1m --gpu 0 --batch_size 512 --epochs 200 --learning_rate 1e-3 --dim 4 --beta 1 --pos_instance 10 --neg_instance 10 --K 30 --D 4 --tau 1.0 --lmbda 0.1
````

#### (Additional) Arguments
````--K````: number of key boxes in each subspace

````--D````: number of subspaces

````--tau````: softmax temperature

````--lmbda````: joint training coefficient
