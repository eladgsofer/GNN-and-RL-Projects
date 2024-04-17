import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
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


class delieveries_dataset():
    def __init__(self, num_of_nodes=50, dataset_size=1000, deliver_probability = 0.95, edge_percentage = 0.2):
        self.num_of_nodes = num_of_nodes
        self.dataset_size = dataset_size
        self.deliver_probability = deliver_probability
        self.edge_percentage = edge_percentage

    def generate_dataset(self,constant_x=True, device='cpu'):
        # generate X(packets) matrix
        X = self.random_matrix(self.num_of_nodes)
        X[np.eye(self.num_of_nodes,dtype=np.bool_)] = 0 # no self connections

        # generate A matrix's
        As = np.zeros([self.dataset_size,self.num_of_nodes,self.num_of_nodes])
        d = np.zeros(self.dataset_size)
        for i in range(self.dataset_size):
            As[i] = self.random_adjacency(self.num_of_nodes,sum=self.deliver_probability,edge_precentage=self.edge_percentage)
            # show_graph_with_labels(As[i])
            d = calculate_deleiveries(As[i],X)
            if i%100 == 0:
                print(i,d)
        print("done")
        self.X      = torch.tensor(X).to(device)
        self.A_list = torch.tensor(As).to(device)
        self.d_list = torch.tensor(d).to(device)
        self.dataset = [0]*self.dataset_size
        for i in range(self.dataset_size):
            rows, cols = np.where(As[i] != 0)
            edges = [[rows.tolist()[i], cols.tolist()[i]] for i in range(len(rows.tolist()))]
            edges = torch.tensor(edges)
            self.dataset[i] = Data(x=X, edge_index=edges.t().contiguous())



    def random_adjacency(self, size, sum, edge_precentage):
        mask_matrix = self.random_matrix(size,sym=True)
        value_matrix = self.random_matrix(size,sym=True)
        # generate valid adjacency matrix
        A = np.zeros([size,size])
        A[mask_matrix<edge_precentage] = value_matrix[mask_matrix<edge_precentage]
        A[np.eye(size,dtype=np.bool_)] = 0 # no self edges
        # make sure the matrix has is connected
        A = self.make_matrix_connected(A)

        # normalize adjacency matrix
        for i in range(6):
            current_sum = A.sum()
            number_of_edges = np.count_nonzero(A)
            target_sum = sum * number_of_edges
            A = np.clip(A, 0,1)
            A = A * (target_sum / current_sum)
        return A
    def make_matrix_connected(self,M):
        mat_size = len(M)
        connected_list = []
        disconnected_list = []
        while len(connected_list) != mat_size:
            binary_map = np.where(M!=0,1,0)+np.eye(mat_size)
            M_v = np.linalg.matrix_power(binary_map,mat_size) # all |V| length paths(which are all the paths)

            # test who is connected to node 0
            connected_to_0 = np.zeros(mat_size)
            connected_to_0[0] = 1
            connected_to_0 = M_v @ connected_to_0 # everyone who is connected to node 0

            connected_list = np.nonzero(connected_to_0)[0]
            disconnected_list = np.where(connected_to_0 == 0)[0]
            if len(disconnected_list) == 0:
                break
            # print(len(connected_list),self.times,self.times2)
            # plt.imshow(M_v==0)
            # plt.show()
            # choose randomly one edge to add between the connected subgraph and the disconnected subgraph
            con_idx = connected_list[np.random.randint(0,len(connected_list))]
            discon_idx = disconnected_list[np.random.randint(0,len(disconnected_list))]
            M[con_idx,discon_idx] = np.random.rand()
            M[discon_idx,con_idx] = M[con_idx,discon_idx]
        return M

    def random_matrix(self,size,sym=False):
        M = np.random.rand(size, size)
        # if we want a symmetric matrix than we simply duplicate the upper right part of the matrix to the lower left part
        if sym:
            M = np.tril(M) + np.triu(M.T, 1)
        return M



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = delieveries_dataset(num_of_nodes=20,dataset_size=1000,edge_percentage=0.2)
    D.generate_dataset(device=device)
    loader = DataLoader(D.dataset, batch_size=32)
    print("success")