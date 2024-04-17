import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import torch
def calculate_deleiveries(A,X):

    best_routes = np.zeros([len(A),len(A)])
    # calculate best routes from every source to every destination
    for i in range(len(A)):
        # The BFS algorithm calculates the best routes from every starting point. It does this jointly for all
        # destination nodes using vectorized notation.
        best_routes[i] = vectorised_BFS(A,i)
    return np.sum(best_routes*X)



def vectorised_BFS(A,start_ind):
    current_routes = np.zeros(len(A))
    current_routes[start_ind] = 1
    pathing_improved = True
    while pathing_improved:
        new_routes = best_one_hop(A, current_routes)
        # accounting for small numerical floating point errors if the new paths are similar to the old path then we have
        # not improved the paths at all, and we can exit the BFS
        if ((current_routes[np.arange(len(A)) != start_ind] - new_routes[np.arange(len(A)) != start_ind])**2).sum() < 10**(-8):
            pathing_improved = False
        current_routes = np.maximum(current_routes, new_routes)
        # print(current_routes)
    return current_routes


def best_one_hop(A,v):
     possiblities_mat = A.T * v
     return np.max(possiblities_mat,axis=1)

# torch to torch
def from_adjacency_tolist(A):
    rows, cols = torch.where(A != 0)
    edges = torch.tensor([[rows.tolist()[i], cols.tolist()[i]] for i in range(len(rows.tolist()))])
    weights = torch.tensor([A[rows, cols][i] for i in range(len(rows.tolist()))])
    return edges, weights


# torch to torch
def from_list_to_adjacency(size, edges,weights):
    A = torch.zeros([size,size])
    for idx,(i,j) in enumerate(edges):
        A[i,j] = weights[idx]
    return A
def show_graph_with_labels(adjacency_matrix):
    rows, cols = np.where(adjacency_matrix != 0)
    weighted_edges = [(rows.tolist()[i], cols.tolist()[i],round(adjacency_matrix[rows,cols].tolist()[i],2))for i in range(len(rows.tolist()))]

    gr = nx.Graph()

    gr.add_weighted_edges_from(weighted_edges)
    pos = nx.spring_layout(gr)
    nx.draw_networkx(gr, pos)
    for edge in gr.edges(data='weight'):
        nx.draw_networkx_edges(gr, pos, edgelist=[edge], width=2*edge[2]**2)
    plt.show()