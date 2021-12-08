import torch
import numpy as np
from nonlinear import run
import utils as ut
import ipdb
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)


#all parameters of the run function in nonlinear.py
n = 5000
d = 10
s0 = 3
graph_type = 'RE'
sem_type = 'mlp'
Mb = 50
#noisemult = 0.6
minibatches_per_NN_training = 250
#clip = '8'
boxpenalty = 0
#methodology = 'plain_vanilla'


#parameters I am playing with in this simulation
methodology_array = ['plain_vanilla','group_clipping','adaclip','adap_quantile','adaclip_and_adap_quantile','group_clipping_and_adap_quantile']
clip_array = ['60', '150 5 8 4', '0', '0', '0', '0']
noisemult_array = [0.6, 0.8, 1.0, 1.2]


#one calc is an average of several tries in order to reduce noise
num_average = 4
def point_est(methodology, clip, noisemult):
    eps_ave = []
    delta_AUC_ROC_ave = []
    delta_AUC_PR_ave = []
    DP_AUC_ROC_ave = []
    DP_AUC_PR_ave = []
    for i in range(num_average):
        ut.set_random_seed(i)
        eps, delta_AUC_ROC, delta_AUC_PR, DP_AUC_ROC, DP_AUC_PR = run(n, d, s0, graph_type, sem_type, Mb, noisemult, minibatches_per_NN_training, clip, boxpenalty, methodology)
        eps_ave.append(eps)
        delta_AUC_ROC_ave.append(delta_AUC_ROC)
        delta_AUC_PR_ave.append(delta_AUC_PR)
        DP_AUC_ROC_ave.append(DP_AUC_ROC)
        DP_AUC_PR_ave.append(DP_AUC_PR)
    return np.average(eps_ave), np.average(delta_AUC_ROC_ave), np.average(delta_AUC_PR_ave), np.average(DP_AUC_ROC_ave), np.average(DP_AUC_PR_ave)


#plotting results
def plot(results,outfile):

    nrows = 1
    ncols = 2
    fig = plt.figure(figsize=(13, 7))

    ax = fig.add_subplot(nrows, ncols, 1)
    legend=[]
    for method, res_method in results.items():
        plt.plot(res_method[0],res_method[3])
        legend.append(method)
    plt.xscale('log')
    plt.title('DP_AUC_ROC')
    plt.ylabel('DP_AUC_ROC')
    plt.xlabel('epsilon')
    plt.legend(legend, loc='best');  

    ax = fig.add_subplot(nrows, ncols, 2)
    legend=[]
    for method, res_method in results.items():
        plt.plot(res_method[0],res_method[1])
        legend.append(method)
    plt.xscale('log')
    plt.title('delta_AUC_ROC')
    plt.ylabel('delta_AUC_ROC')
    plt.xlabel('epsilon')
    plt.legend(legend, loc='best');  

    plt.savefig(outfile, bbox_inches='tight')
    

#saving results in a file
def save_res(results, filename):
    f = open(filename, 'w')
    f.write("Samples " + str(n) + " Graph " + str(d) + " " + str(s0) + " " + graph_type + " " + sem_type + " Minibatch size " + str(Mb) + " Number of minibatches " +
        str(minibatches_per_NN_training) + " Box penalty " + str(boxpenalty) + "\n")
    f.write(str(results) + "\n")
    f.close()


if __name__ == '__main__':

    results = {}
    for a,b in zip(methodology_array,clip_array):
        results[a]=[[],[],[],[],[]]
        for c in noisemult_array:
            res = point_est(a,b,c)
            results[a][0].append(res[0])
            results[a][1].append(res[1])
            results[a][2].append(res[2])
            results[a][3].append(res[3])
            results[a][4].append(res[4])
            
    #ipdb.set_trace()

    print(results)
    plot(results, './outputs/simulation2.png')
    save_res(results, './outputs/simulation2.txt')


