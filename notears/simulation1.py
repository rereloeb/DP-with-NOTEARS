import torch
import numpy as np
from nonlinear import run
import utils as ut
import ipdb
import matplotlib.pyplot as plt


torch.set_default_dtype(torch.double)
torch.set_printoptions(precision=3)
np.set_printoptions(precision=3)


n = 5000
d = 10
s0 = 2
graph_type = 'RE'
sem_type = 'mim'
Mb = 50
#noisemult = 0.6
minibatches_per_NN_training = 250
#clip = '8'
boxpenalty = 0
#methodology = 'plain_vanilla'


num_average = 2
def point_est(methodology, clip, noisemult):
    eps_ave = []
    delta_AUC_ROC_ave = []
    delta_AUC_PR_ave = []
    for i in range(num_average):
        ut.set_random_seed(i)
        eps, delta_AUC_ROC, delta_AUC_PR = run(n, d, s0, graph_type, sem_type, Mb, noisemult, minibatches_per_NN_training, clip, boxpenalty, methodology)
        eps_ave.append(eps)
        delta_AUC_ROC_ave.append(delta_AUC_ROC)
        delta_AUC_PR_ave.append(delta_AUC_PR)
    return np.average(eps_ave), np.average(delta_AUC_ROC_ave), np.average(delta_AUC_PR_ave)


#methodology_array = ['plain_vanilla','group_clipping','adaclip','adap_quantile','adaclip_and_adap_quantile','group_clipping_and_adap_quantile']
#clip_array = ['8', '3.1 1 6 3', '0', '0', '0', '0']
#noisemult_array = [0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
methodology_array = ['plain_vanilla', 'adap_quantile']
clip_array = ['8', '0']
noisemult_array = [0.1, 0.5, 0.9]


results = {}
for a,b in zip(methodology_array,clip_array):
    results[a]=[[],[],[]]
    for c in noisemult_array:
        res = point_est(a,b,c)
        results[a][0].append(res[0])
        results[a][1].append(res[1])
        results[a][2].append(res[2])
ipdb.set_trace()
print(results)


