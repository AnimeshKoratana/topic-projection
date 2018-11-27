import torch
import numpy as np
import graph
import eval

r'''
calculation of pairwise distance, and return condensed result, i.e. we omit the diagonal and duplicate entries and store everything in a one-dimensional array
'''
def pairwise_distance(data1, data2=None, device=-1):
    r'''
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    '''
    if data2 is None:
        data2 = data1

    if device!=-1:
        data1, data2 = data1.cuda(device), data2.cuda(device)

    #N*1*M
    A = data1.unsqueeze(dim=1)

    #1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A-B)**2.0
    #return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis

def group_pairwise(X, groups, device=0, fun=lambda r,c: pairwise_distance(r, c).cpu()):
    group_dict = {}
    for group_index_r, group_r in enumerate(groups):
        for group_index_c, group_c in enumerate(groups):
            R, C = X[group_r], X[group_c]
            if device!=-1:
                R = R.cuda(device)
                C = C.cuda(device)
            group_dict[(group_index_r, group_index_c)] = fun(R, C)
    return group_dict


def forgy(X, n_clusters):
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def lloyd(X, n_clusters, graph, device=0, tol=1e-4):
    X = torch.from_numpy(X).float().cuda(device)

    initial_state = forgy(X, n_clusters)

    num_iterations = 0
    while num_iterations < 30:
        dis = pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster==index).squeeze()

            selected = torch.index_select(X, 0, selected)
            if len(selected) == 0:
                m = torch.Tensor([0])
            else:
                m = selected.mean(dim=0)
            initial_state[index] = m


        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        num_iterations += 1
        #calculate mean_distances
        # print(eval.davies_bouldin(graph, choice_cluster.cpu().numpy(), initial_state))

        if center_shift ** 2 < tol:
            break

    return choice_cluster.cpu().numpy(), initial_state.cpu().numpy()

def main():
    sizes = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    ks = [5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]

    distances = []
    for i, s in enumerate(sizes):
        z = []
        g = graph.Graph(num_pages=s)
        x = [p.embedding() for p in g]
        x = np.stack(x)
        for j, k in enumerate(ks):
            print("Running size {}, k={}".format(s, k))
            design_matrix = x.copy()
            choice_cluster, initial_state = lloyd(design_matrix, k, g)
            d = eval.davies_bouldin(g, choice_cluster, initial_state, k)
            z.append(d)
        distances.append(z)
    all = ""
    for i, _ in enumerate(sizes):
        s = ""
        for j, _ in enumerate(ks):
            s = s + (str(distances[i][j]) + "\t")
        all  = all + s + "\n"
    with open("out.txt", "w+") as w:
        w.write(all)



if __name__ == '__main__':
    main()
