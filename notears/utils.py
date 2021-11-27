import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
import ipdb
import torch


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)
        B_und = _graph_to_adjmat(G_und)
        #print("undirected graph",B_und)
        B = _random_acyclic_orientation(B_und)
        #print("DAG",B)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)), directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = int(0.2 * d)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    elif  graph_type == 'RE':
        # Generates a k-regular random graph (both the in-degree and the out-degree of each vertex will be k before enforcing acyclicity)
        G_und = ig.Graph.K_Regular(n=d, k=s0, directed=True)        
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
    """Simulate SEM parameters for a DAG.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        w_ranges (tuple): disjoint weight ranges

    Returns:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
    """
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U
    return W



def simulate_nonlinear_sem(B, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM.

    Args:
        B (np.ndarray): [d, d] binary adj matrix of DAG
        n (int): num of samples
        sem_type (str): mlp, mim, gp, gp-add
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, scale):
        """X: [n, num of parents], x: [n]"""
        z = np.random.normal(scale=scale, size=n)
        pa_size = X.shape[1]
        if pa_size == 0:
            return z
        if sem_type == 'mlp':
            hidden = 100
            W1 = np.random.uniform(low=0.5, high=2.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.5, high=2.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            x = sigmoid(X @ W1) @ W2 + z
        elif sem_type == 'mim':
            w1 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w1[np.random.rand(pa_size) < 0.5] *= -1
            w2 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w2[np.random.rand(pa_size) < 0.5] *= -1
            w3 = np.random.uniform(low=0.5, high=2.0, size=pa_size)
            w3[np.random.rand(pa_size) < 0.5] *= -1
            x = np.tanh(X @ w1) + np.cos(X @ w2) + np.sin(X @ w3) + z
        elif sem_type == 'gp':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = gp.sample_y(X, random_state=None).flatten() + z
        elif sem_type == 'gp-add':
            from sklearn.gaussian_process import GaussianProcessRegressor
            gp = GaussianProcessRegressor()
            x = sum([gp.sample_y(X[:, i, None], random_state=None).flatten()
                     for i in range(X.shape[1])]) + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = B.shape[0]
    scale_vec = noise_scale if noise_scale else np.ones(d)
    X = np.zeros([n, d])
    G = ig.Graph.Adjacency(B.tolist())
    ordered_vertices = G.topological_sorting()
    #print("B",B,"list",B.tolist(),"G",G,"order",ordered_vertices,"in-degrees",G.degree(mode="in"))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], scale_vec[j])
    return X


def count_accuracy(B_true, B_est):

    """Compute various accuracy metrics for B_est.

    true positive = predicted association exists in condition in correct direction
    reverse = predicted association exists in condition in opposite direction
    false positive = predicted association does not exist in condition

    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B_est (np.ndarray): [d, d] estimate, {0, 1, -1}, -1 is undirected edge in CPDAG

    Returns:
        fdr: (reverse + false positive) / prediction positive
        tpr: (true positive) / condition positive
        fpr: (reverse + false positive) / condition negative
        shd: undirected extra + undirected missing + reverse
        nnz: prediction positive
    """

    #if (B_est == -1).any():  # cpdag
        #if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            #raise ValueError('B_est should take value in {0,1,-1}')
        #if ((B_est == -1) & (B_est.T == -1)).any():
            #raise ValueError('undirected edge should only appear once')
    #else:  # dag
        #if not ((B_est == 0) | (B_est == 1)).all():
            #raise ValueError('B_est should take value in {0,1}')
        #if not is_dag(B_est):
            #raise ValueError('B_est should be a DAG')

    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    if 1-fdr+tpr == 0:
        f1 = 'NA'
    else:
        f1 = 2*(1-fdr)*tpr/(1-fdr+tpr)

    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'f1': f1, 'shd': shd, 'npred': pred_size, 'ntrue': len(cond)}


def auc(B_true, B):
    """Uses count_accuracy() to calculate aucroc and aucpr
    
    Args:
        B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
        B (np.ndarray): [d, d] output of the algorithm before thresholding

    Returns:
        aucroc
        aucpr
    """
    #ipdb.set_trace()
    C = B[~np.eye(B.shape[0],dtype=bool)].reshape(B.shape[0],-1)
    C = C.flatten()
    print(C)
    print(B_true)
    C_true = B_true[~np.eye(B_true.shape[0],dtype=bool)].reshape(B_true.shape[0],-1)
    C_true = C_true.flatten()
    print(C_true)
    return roc_auc_score(C_true, C), average_precision_score(C_true, C)


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def show_stats(model, X, boxpenalty, lambda1, lambda2, lambda3, rho, alpha, B_true):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    with torch.no_grad():
        X_torch = torch.from_numpy(X)
        X_torch = X_torch.to(device)
        Xhat2 = model(X_torch)
        h_val = model.h_func()
        #ipdb.set_trace()
        loss = squared_loss(Xhat2, X_torch)
        penalty = 0.5 * rho * h_val * h_val + alpha * h_val
        penalty1 = 0.5 * rho * h_val * h_val
        penalty2 = alpha * h_val
        l2_reg = 0.5 * lambda2 * model.l2_reg()
        l1_reg = lambda1 * model.fc1_l1_reg()
        if boxpenalty:
            bound_pen = lambda3 * model.bound_penalty()
            globalloss = loss + penalty + l2_reg + l1_reg + bound_pen
        else:
            globalloss = loss + penalty + l2_reg + l1_reg
            
        W_est = model.fc1_to_adj()
        auc_DP = auc(B_true, W_est)
        print("aucroc, aucpr",auc_DP)

        W_est[np.abs(W_est) < 0.3] = 0
        shd = count_accuracy(B_true, W_est != 0)['shd']
        DAG = is_dag(W_est)
        if boxpenalty:
            print( "Objective function " + "{:.2f}".format(globalloss.item()) +
                " = squared loss an data " + "{:.2f}".format(loss.item()) + " + 0.5*rho*h**2 " + "{:.6f}".format(penalty1.item()) +
                " + alpha*h " + "{:.6f}".format(penalty2.item()) + " + L2reg " + "{:.2f}".format(l2_reg.item()) +
                " + L1reg "+ "{:.2f}".format(l1_reg.item()) + " + box penalty " + "{:.2f}".format(bound_pen.item()) +
                " ; SHD = " + str(shd) + " ; DAG " + str(DAG) )
        else:
            print( "Objective function " + "{:.2f}".format(globalloss.item()) +
            " = squared loss an data " + "{:.2f}".format(loss.item()) + " + 0.5*rho*h**2 " + "{:.6f}".format(penalty1.item()) +
            " + alpha*h " + "{:.6f}".format(penalty2.item()) + " + L2reg " + "{:.2f}".format(l2_reg.item()) +
#            " + L1reg "+ "{:.2f}".format(l1_reg.item()) + " + box penalty " + "{:.2f}".format(bound_pen.item()) +
            " + L1reg "+ "{:.2f}".format(l1_reg.item()) +
            " ; SHD = " + str(shd) + " ; DAG " + str(DAG) )


if __name__ == '__main__':

#    C      = [ 5.167, 4.123 ,3.999 , 3.765 , 2.754, 2.123 , 1.973 ,1.654 , 1.314 , 0.923, 0.456, 0.234 ]

#    C_true = [ 1.   , 0.   ,1.    , 0.   , 1.   , 0.   , 1.   , 0.   , 1.   , 0.   , 1.   , 0.   ]

#    fpr, tpr, thresholds = roc_curve(C_true, C)
#    print(fpr)
#    print(tpr)
#    print(thresholds)
#    print(roc_auc_score(C_true, C))

#    precision, recall, thresholds = precision_recall_curve(C_true, C)
#    print(recall)
#    print(precision)
#    print(thresholds)
#    print(average_precision_score(C_true, C))

    n = 5000
    B_true = simulate_dag(10, 6, 'RE')
    print(B_true)
    X = simulate_nonlinear_sem(B_true, n, 'mlp')
    maxdegree = B_true.sum(axis=0).max()
    complexity1 = n/(2**(maxdegree+1))
    degrees = B_true.sum(axis=0)
    powers = [2**(de+1) for de in degrees]
    complexity2 = n / ( sum(powers) / len(powers) )
    print("max degree ",maxdegree, " complexity indicator 1 ",complexity1," complexity indicator 2 ",complexity2)


