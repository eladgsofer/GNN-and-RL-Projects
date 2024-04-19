from dataset import delieveries_dataset
import torch
from matplotlib import pyplot as plt
from utils import show_graph_with_labels

def vectorized_floyd_warshall(A,device):
    mat_size = len(A)
    # initialize pathings
    output_paths = torch.fill(torch.zeros([mat_size, mat_size]),torch.nan).to(device)
    for i in range(mat_size):
        for j in range(mat_size):
            if A[i,j]!=0 or i==j:
                output_paths[i,j] = int(i)
    # initialize lengths
    output_lengths = A.detach().clone()
    output_lengths.fill_diagonal_(1)
    for k in range(mat_size):
        best_paths_for_subgraph_k = output_lengths[:,k].unsqueeze(1)*output_lengths[k,:].unsqueeze(0)
        # set pathings
        mask = (best_paths_for_subgraph_k-output_lengths)>10**(-9)
        new_paths = output_paths[k,:]
        new_paths_mat = new_paths.repeat(mat_size,1)
        output_paths[mask] = new_paths_mat[mask]
        # set new lengths
        output_lengths = torch.maximum(output_lengths,best_paths_for_subgraph_k)

    return output_paths.type(torch.int), output_lengths
def get_path(pathings_mat,start_ind,end_ind):
    if pathings_mat[start_ind,end_ind] == torch.nan:
        return []
    path = [end_ind]
    iter = 0
    while end_ind != start_ind:
        end_ind = pathings_mat[start_ind,end_ind]
        # print(start_ind,end_ind)
        path.append(end_ind)
        iter = iter+1
        # if iter > 30:
        #     return -1,path
    return 0,path

def floyd_warshall_internet(A): # from geeks to geeks
    A = A.detach().clone()
    V= len(A)
    for k in range(V):

        # pick all vertices as source one by one
        for i in range(V):

            # Pick all vertices as destination for the
            # above picked source
            for j in range(V):
                # If vertex k is on the shortest path from
                # i to j, then update the value of dist[i][j]
                A[i][j] = max(A[i][j],
                                 A[i][k] * A[k][j]
                                 )
    return A
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    D = delieveries_dataset(num_of_nodes=20, dataset_size=100, edge_percentage=0.05)
    D.generate_dataset(device=device)
    mat_shape = D.A_list[0].shape

    output_paths, output_lengths = vectorized_floyd_warshall(D.A_list[0],device)
    test_algorithm = floyd_warshall_internet(D.A_list[0])
    print((output_lengths*D.X).sum())
    print((test_algorithm*D.X).sum())
    for m in range(50):
        print("new matrix")
        output_paths, output_lengths = vectorized_floyd_warshall(D.A_list[m], device)
        for i in range(16):
            for j in range(16):
                print("starting again")
                s,t = get_path(output_paths, i, j)
                if s==-1:
                    show_graph_with_labels(D.A_list[m].cpu().numpy())
                    print(t)
                    output_paths, output_lengths = vectorized_floyd_warshall(D.A_list[m], device)
                print(t)
    print(D.d_list[0])