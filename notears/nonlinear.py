from locally_connected import LocallyConnected
from lbfgsb_scipy3 import LBFGSBScipy
from trace_expm import trace_expm
from model_creation import NotearsMLP

import torch
import torch.nn as nn
import numpy as np
import math
import DPoptimizer as dpopt
import ipdb
import epsilon_calculation
import utils as ut
import batch_samplers
import argparse


def squared_loss(output, target):
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def show_params_and_gradients_and_h(model):
    for f in model.parameters():
        print('Param is',f.data)
        #print('Gradient is ',f.grad)
    print('h(A)',model.h_func().item())
    print(model.fc1_pos.weight - model.fc1_neg.weight)


def dual_ascent_step(model, X, boxpenalty, lambda1, lambda2, lambda3, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None

    optimizer = LBFGSBScipy(model.parameters(), boxpenalty)
#cf. maxiter in LBFGSBScipy (optiondict) that is 15000 by default !
#cf. maxcor in LBFGSBScipy (optiondict)
#should it be compared to the number of epochs or number of batches ( = number of updates) in the stochastic gradient descent methods ?

#torch LFBGS optimizer
    #optimizer = torch.optim.LBFGS( model.parameters(), lr=0.05, max_iter=1000, history_size=20 )
#lr = 0.01 0.05 ? other possible method: line_search_fn = 'strong_wolfe'

    k=1
    X_torch = torch.from_numpy(X)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    X_torch.to(device)

    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
#in the case of box penalty, there is no need for bound_pen in the LBFGSBScipy optimizer, it has bounds embedded unlike the torch one (that needs bound_pen consequently)
            #bound_pen = lambda3 * model.bound_penalty()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        print("iteration "+str(k)+" in inner loop,alpha "+str(alpha)+" rho "+str(rho)+" h "+str(h_new))
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
        k += 1
    alpha += rho * h_new
    return rho, alpha, h_new


#microbatch size, delta
mb = 1
delta = 1e-5


def DP_dual_ascent_step(model, X, boxpenalty, method, Mb, noisemult, minibatches_per_NN_training, clip, lambda1, lambda2, lambda3, rho, alpha, h, rho_max, iterations, B_true):

    h_new = None
    kk=1
    X_torch = torch.utils.data.TensorDataset ( torch.from_numpy(X) )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

#DP Adam optimizer creation
    learnrate = 0.01 #for plain_vanilla, group_clipping and group_clipping_and_adap_quantile, lr was 0.001 and optimizer was here before merging codes
    #DPAdam = dpopt.make_optimizer_class(torch.optim.Adam, method)
    #optimizer = DPAdam(params=model.parameters(), lr=learnrate, l2_norm_clip=clip, noise_multiplier=noisemult, minibatch_size=Mb, microbatch_size=mb)

#initialization of the scheduler
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( optimizer, patience=5000, verbose = True )
    #lamb = lambda iterations: 0.1 if iterations % 10000 > 5000 else 1.0
    #scheduler = torch.optim.lr_scheduler.LambdaLR( optimizer, lr_lambda = lamb, verbose = False )

    while rho < rho_max:

        DPAdam = dpopt.make_optimizer_class(torch.optim.Adam, method)
        optimizer = DPAdam(params=model.parameters(), lr=learnrate, l2_norm_clip=clip, noise_multiplier=noisemult, minibatch_size=Mb, microbatch_size=mb)

        minibatch_loader, microbatch_loader = batch_samplers.get_data_loaders( Mb, mb, minibatches_per_NN_training )
        ut.show_stats(model, X, boxpenalty, lambda1, lambda2, lambda3, rho, alpha, B_true)

        for x_batch in minibatch_loader( X_torch ):
            x_batch = x_batch[0]
            optimizer.zero_grad()
            for X_microbatch in microbatch_loader( torch.utils.data.TensorDataset(x_batch) ):
                X_microbatch = X_microbatch[0]
                X_microbatch = X_microbatch.to(device)
                optimizer.zero_microbatch_grad()
                X_hat = model(X_microbatch)
                loss = squared_loss(X_hat, X_microbatch)
                h_val = model.h_func()
                penalty = 0.5 * rho * h_val * h_val + alpha * h_val
                l2_reg = 0.5 * lambda2 * model.l2_reg()
                l1_reg = lambda1 * model.fc1_l1_reg()
                if boxpenalty:
                    bound_pen = lambda3 * model.bound_penalty()
                    primal_obj = loss + penalty + l2_reg + l1_reg + bound_pen
                else:
                    primal_obj = loss + penalty + l2_reg + l1_reg
                primal_obj.backward()
                optimizer.microbatch_step()
            optimizer.step()

#scheduler step
            #x_hat = model(x_batch)
            #loss = squared_loss(x_hat, x_batch)
            #h_val = model.h_func()
            #penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            #l2_reg = 0.5 * lambda2 * model.l2_reg()
            #l1_reg = lambda1 * model.fc1_l1_reg()
            #bound_pen = lambda3 * model.bound_penalty()
            #training_loss_minibatch = loss + penalty + l2_reg + l1_reg + bound_pen
            #scheduler.step( training_loss_minibatch )
            #scheduler.step()

            iterations += 1

        ut.show_stats(model, X, boxpenalty, lambda1, lambda2, lambda3, rho, alpha, B_true)

        if (method == 'group_clipping') or (method == 'group_clipping_and_adap_quantile'):
            for group in optimizer.param_groups:
                prop = [ x.item() / y.item() for x, y in zip(group['nbclip'], group['nbmb']) ]
                print("Proportion of microbatches that were clipped ", prop)
        else:
            print("Proportion of microbatches that were clipped ", optimizer.nbclip / optimizer.nbmb )

        with torch.no_grad():
            h_new = model.h_func().item()
        print("iteration "+str(kk)+" in inner loop, alpha "+str(alpha)+" rho "+str(rho)+" h "+str(h_new))

        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
        kk += 1

    alpha += rho * h_new

    return rho, alpha, h_new, iterations


def notears_nonlinear(B_true, model: nn.Module, X: np.ndarray, boxpenalty: bool, method: str, Mb: int = 0, noisemult: float = 0.0, minibatches_per_NN_training: int = 0,
    clip=[[0,0,0,0,0,0]], lambda1: float = 0., lambda2: float = 0., lambda3: float = 0., max_iter: int = 100, h_tol: float = 1e-8,
    rho_max: float = 1e+6, w_threshold: float = 0.3, DPflag: bool = False):

    rho, alpha, h, iterations = 1.0, 0.0, np.inf, 0

    for _ in range(1,max_iter+1):
        if not DPflag:
            rho, alpha, h = dual_ascent_step(model, X, boxpenalty, lambda1, lambda2, lambda3, rho, alpha, h, rho_max)
        else:
            rho, alpha, h, iterations = DP_dual_ascent_step(model, X, boxpenalty, method, Mb, noisemult, minibatches_per_NN_training, clip,
                lambda1, lambda2, lambda3, rho, alpha, h, rho_max, iterations, B_true)
        print("iteration "+str(_)+" in outer loop, alpha = "+str(alpha)+", rho = "+str(rho)+", h = "+str(h))
        if h <= h_tol or rho >= rho_max:
            break

    W_est = model.fc1_to_adj()
    W = W_est.copy()
    print('Threshold',w_threshold)
    W_est[np.abs(W_est) < w_threshold] = 0
    with np.printoptions(precision=3, suppress=True):
        print(W)
        print(W_est)

    return W_est , iterations, W


def run(n, d, s0, graph_type, sem_type, Mb, noisemult, minibatches_per_NN_training, clip, boxpenalty, methodology):

    print("samples ",n," SCM ",d,s0,graph_type,sem_type," minibatch size ",Mb," noise ",noisemult," minibatches per NN training ",minibatches_per_NN_training,
        " DP methodology ",methodology, clip," box penalty ", boxpenalty)

    if methodology == 'group_clipping':
#to be recoded if models get more complicated (more groups in param_groups)
        clip = [[ float(item) for item in clip.split() ]]
    else:
        clip = float(clip)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

#generation of the SCM and the dataset
    B_true = ut.simulate_dag(d, s0, graph_type)
    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    maxdegree = B_true.sum(axis=0).max()
    complexity1 = n/(2**(maxdegree+1))
    degrees = B_true.sum(axis=0)
    powers = [2**(de+1) for de in degrees]
    complexity2 = n / ( sum(powers) / len(powers) )
    complexity3 = n / ( sum(powers) )
    print("max degree ",maxdegree, " complexity indicator 1 ",complexity1," complexity indicator 2 ",complexity2," complexity indicator 3 ",complexity3)

#non private causal search
    model = NotearsMLP(dims=[d, 10, 1], boxpenalty=boxpenalty, bias=True)
    model.to(device)
    W_est , iterations, W = notears_nonlinear(B_true, model, X, boxpenalty, methodology, lambda1=0.01*boxpenalty+0.001*(1-boxpenalty), lambda2=0.01, lambda3=10.0)
    assert ut.is_dag(W_est)
    acc = ut.count_accuracy(B_true, W_est != 0)
    print(acc)
    auc = ut.auc(B_true, W)
    print("aucroc, aucpr",auc)

#DP causal search
    model = NotearsMLP(dims=[d, 10, 1], boxpenalty=boxpenalty, bias=True)
    model.to(device)
    
    if (methodology == 'group_clipping') or (methodology == 'group_clipping_and_adap_quantile'):
#RECODE THAT
        if boxpenalty==0:
            G = 4
        else:
            G = 6
        noisemult_modified = noisemult * math.sqrt(G)
        print("number of groups for clipping ",G," noise multiplier modified for group clipping ",noisemult_modified)
        W_est_DP , iterations, W = notears_nonlinear(B_true, model, X, boxpenalty, methodology, Mb, noisemult_modified, minibatches_per_NN_training, clip,
            lambda1=0.01*boxpenalty+0.001*(1-boxpenalty), lambda2=0.01, lambda3=1e+2, DPflag=True)
    else:
        W_est_DP , iterations, W = notears_nonlinear(B_true, model, X, boxpenalty, methodology, Mb, noisemult, minibatches_per_NN_training, clip,
            lambda1=0.01*boxpenalty+0.001*(1-boxpenalty), lambda2=0.01, lambda3=1e+2, DPflag=True)
            
    assert ut.is_dag(W_est_DP)
    acc_DP = ut.count_accuracy(B_true, W_est_DP != 0)
    print(acc_DP)
    auc_DP = ut.auc(B_true, W)
    print("aucroc, aucpr",auc_DP)

#privacy budget calculation
    #ipdb.set_trace()
    print("Iterations",iterations)
    eps = epsilon_calculation.epsilon(n, Mb, noisemult, iterations, delta)
    print( 'Achieves ({}, {})-DP'.format(eps,delta) )
    #show_params_and_gradients_and_h(model)
    #for name, param in model.named_parameters():
        #if param.requires_grad:
            #print (name, param.data)

    return eps, auc[0]-auc_DP[0], auc[1]-auc_DP[1], auc_DP[0], auc_DP[1]


if __name__ == '__main__':

#precisions, displays and seeds
    torch.set_default_dtype(torch.double)
    torch.set_printoptions(precision=3)
    np.set_printoptions(precision=3)
    ut.set_random_seed(55)

#parsing of the arguments
    parser = argparse.ArgumentParser(description='FCM parameters')
    parser.add_argument('--samples', type=int, help='Number of samples', required = True)
    parser.add_argument('--nodes', type=int, help='Number of nodes / variables', required = True)
    parser.add_argument('--edges', type=int, help='Number of edges or max degree for RE graph type', required = True)
    parser.add_argument('--graphtype', choices = ['ER','SF','BP','RE'], help='Graph type ER or SF or BP or RE', required = True)
    parser.add_argument('--SEMtype', choices = ['mlp','mim','gp','gp-add'], help='SEM type mlp or mim or gp or gp-add', required = True)
    parser.add_argument('--minibatch', type=int, help='Minibatch size', required = True)
    parser.add_argument('--noisemult', type=float, help='Noise multiplier', required = True)
    parser.add_argument('--minibatchesperNNtraining', type=int, help='Number of minibatches used for each NN training', required = True)
    parser.add_argument('--clip', type=str, help='Clipping thresholds per param group, used for plain_vanilla and group_clipping methodologies', required = True)
    parser.add_argument('--boxpenalty', type=int, choices = [0,1], help='if 1 first layer is expressed as the diff between 2 positive layers. otherwise has to be 0', required = True)
    parser.add_argument('--method', choices = ['plain_vanilla','group_clipping','adaclip','adap_quantile','adaclip_and_adap_quantile','group_clipping_and_adap_quantile'],
        help='DP optimizer methodology', required = True)
    args = parser.parse_args()

    n, d, s0, graph_type, sem_type = args.samples, args.nodes, args.edges, args.graphtype, args.SEMtype
    Mb, noisemult, minibatches_per_NN_training, clip = args.minibatch, args.noisemult, args.minibatchesperNNtraining, args.clip
    boxpenalty = args.boxpenalty
    methodology = args.method

    run(n, d, s0, graph_type, sem_type, Mb, noisemult, minibatches_per_NN_training, clip, boxpenalty, methodology)
    
    
