import os
import math
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from fastdtw import fastdtw
from .utils import log_string
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

def laplacian(W):
    d = W.sum(axis=0)
    d_inv_sqrt = np.power(d, -0.5, where=(d != 0))
    d_inv_sqrt[d == 0] = 0.0
    D = sp.diags(d_inv_sqrt, 0)
    I = sp.identity(d.size, dtype=W.dtype)
    L = I - D @ W @ D
    return L
def largest_k_lamb(L, k):
    lamb, U = sp.linalg.eigsh(L, k=k, which='LM')
    return lamb, U
def get_eigv(adj,k):
    L = laplacian(adj)
    eig = largest_k_lamb(L,k)
    return eig
def construct_tem_adj(data, num_node, steps_per_day=288):
    num_days = data.shape[0] // steps_per_day
    data_mean = np.mean([data[steps_per_day * i: steps_per_day * (i + 1)] for i in range(num_days)], axis=0)
    data_mean = data_mean.squeeze().T
    dtw_distance = np.zeros((num_node, num_node))
    for i in tqdm(range(num_node), desc="DTW Calculation"):
        for j in range(i + 1, num_node):
            dtw_distance[i][j] = fastdtw(data_mean[i], data_mean[j], radius=6)[0]
            dtw_distance[j][i] = dtw_distance[i][j]
    num_edges = int(np.log2(num_node) * num_node)
    nth = np.partition(dtw_distance.flatten(), num_edges)[num_edges]
    tem_matrix = np.zeros_like(dtw_distance)
    tem_matrix[dtw_distance <= nth] = 1
    np.fill_diagonal(tem_matrix, 0)
    tem_matrix = np.logical_or(tem_matrix, tem_matrix.T).astype(int)
    return tem_matrix
def load_graph(spatial_graph, temporal_graph, dims, data, log):
    adj = np.load(spatial_graph, allow_pickle=True)
    adj = adj + np.eye(adj.shape[0])
    if os.path.exists(temporal_graph):
        tem_adj = np.load(temporal_graph)
    else:
        tem_adj = construct_tem_adj(data, adj.shape[0])
        np.save(temporal_graph, tem_adj)
    spa_wave = get_eigv(adj, dims)
    tem_wave = get_eigv(tem_adj, dims)
    log_string(log, f'Shape of graph wave eigenvalue and eigenvector: {spa_wave[0].shape}, {spa_wave[1].shape}')

    sampled_nodes_number = int(math.log(adj.shape[0], 2))
    graph = csr_matrix(adj)
    dist_matrix = dijkstra(csgraph=graph)
    dist_matrix[dist_matrix==0] = dist_matrix.max() + 10
    local_adj = np.argpartition(dist_matrix, sampled_nodes_number-1,axis=-1)[:, :sampled_nodes_number]

    log_string(log, f'Shape of local_adj: {local_adj.shape}')
    return local_adj, spa_wave, tem_wave