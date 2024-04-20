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
        mask = (best_paths_for_subgraph_k-output_lengths)>10**(-7)
        new_paths = output_paths[k,:]
        new_paths_mat = new_paths.repeat(mat_size,1)
        output_paths[mask] = new_paths_mat[mask]
        if torch.any(output_paths<0):
            print("negative pathing!")
        # set new lengths
        output_lengths = torch.maximum(output_lengths,best_paths_for_subgraph_k)

    return output_paths.type(torch.int), output_lengths
def get_path(pathings_mat,start_ind,end_ind,device):
    if pathings_mat[start_ind,end_ind] == torch.nan:
        return []
    path = [torch.tensor(end_ind,device=device,dtype=torch.int)]
    iter = 0
    while end_ind != start_ind:
        if end_ind<0:
            print("negative?")
        end_ind = pathings_mat[start_ind,end_ind]
        # print(start_ind,end_ind)
        path.append(end_ind)
        iter = iter+1
        if iter > len(pathings_mat):
            print("error path too long...")
            exit()
            # return path
    return path

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
def best_pathings(A,paths_mat,device):
    mat_size = len(A)
    pathings_tensor = torch.ones([mat_size,mat_size,mat_size],device=device)
    for start_ind in range(mat_size):
        for end_ind in range(mat_size):
            print(start_ind,end_ind)
            if start_ind == end_ind:
                continue
            path = get_path(paths_mat,start_ind,end_ind,device)
            last_node = path[-1]
            path = path[:-1]
            for step,current_node in enumerate(path[::-1]):
                pathings_tensor[start_ind,end_ind][step] = A[last_node,current_node]
    return pathings_tensor
def delivery_loss(A,paths_mat,X,device):
    mat_size = len(A)
    loss_sum = torch.zeros(1,device=device)

    for start_ind in range(mat_size):
        for end_ind in range(mat_size):
            if start_ind == end_ind:
                continue
            loss_prod = torch.ones(1,device=device)
            path = get_path(paths_mat, start_ind, end_ind, device)
            last_node = path[-1]
            path = path[:-1]
            for step, current_node in enumerate(path[::-1]):
                loss_prod = loss_prod* A[last_node, current_node]
                last_node = current_node

            loss_sum = loss_sum+ loss_prod*X[start_ind,end_ind]
    return loss_sum
def projection(A,target_sum,epsilon):
    if target_sum<torch.sum(A):
        A = target_sum*(torch.abs(A)/torch.sum(A))
    return torch.clip(A,0,1-epsilon)
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    D = delieveries_dataset(num_of_nodes=25, dataset_size=100, edge_percentage=0.15,deliver_probability=0.6)
    D.generate_dataset(device=device)
    mat_shape = D.A_list[0].shape
    output_paths, output_lengths = vectorized_floyd_warshall(D.A_list[0],device)
    test_algorithm = floyd_warshall_internet(D.A_list[0])
    print("floyd warshall initial ",(output_lengths*D.X).sum())
    print("floyd warshall internet validation",(test_algorithm*D.X).sum())
    A_with_grad = D.A_list[0]
    iterations = 10
    optimization_iters = 10

    for i in range(iterations):
        print("optimizing paths")
        A = A_with_grad.detach().clone()
        show_graph_with_labels(A)
        output_paths, deliveries = vectorized_floyd_warshall(A,device)
        print("reoptimizing paths, new paths deliveries",(deliveries * D.X).sum().item())
        A_with_grad = A.clone().detach().requires_grad_(True).to(device)
        # pathings_tensor = best_pathings(A,output_paths,device)
        print("new pathings")
        # optimizer
        # optimizer = torch.optim.Adam([A_with_grad],lr=0.01)
        torch.autograd.set_detect_anomaly(True)
        for j in range(optimization_iters):
            A_with_grad = A_with_grad.clone().detach().requires_grad_(True).to(device)

            # loss
            loss = delivery_loss(A_with_grad,output_paths,D.X,device)
            grad = torch.autograd.grad(loss,A_with_grad)[0]
            A_with_grad = A_with_grad + 0.001*grad

            A_with_grad = projection(A_with_grad,D.deliver_probability*torch.count_nonzero(A_with_grad),D.epsilon)


            _, deliveries = vectorized_floyd_warshall(A_with_grad.detach().clone(), device)
            print((deliveries*D.X).sum().item(),loss.item())

    print(D.d_list[0])