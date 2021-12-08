

    python nonlinear.py --samples 5000 --nodes 50 --edges 2 --graphtype 'RE' --SEMtype 'mlp' \
    --minibatch 100 --noisemult 0.01 --minibatchesperNNtraining 500 --clip '0' --boxpenalty 0 --method 'adap_quantile' \


# minibatches per NN training = 100, Mb = 100, noise = 0.01, we got (79,64) w/o box penalty
# minibatches per NN training = 500, Mb = 100, noise = 0.01, we got (?,?) w/o box penalty
# minibatches per NN training = 100, Mb = 100, noise = 0.01, we got (48,15) w/o box penalty modified lr to 0.001 instead of 0.01

# minibatches per NN training = 1000, Mb = 50, noise = 0.1, we got (79,37) w/o box penalty
# minibatches per NN training = 500, Mb = 50, noise = 0.1, we got (77,38) w/o box penalty
# minibatches per NN training = 250, Mb = 50, noise = 0.1, we got (75,46) w/o box penalty
# minibatches per NN training = 250, Mb = 50, noise = 0.1, we got (70,43) w box penalty
# minibatches per NN training = 100, Mb = 100, noise = 0.1, we got (83,51) w/o box penalty
# minibatches per NN training = 250, Mb = 50, noise = 0.1, we got (74,40) w/o box penalty modified quantile to 10% instead of 25%

# minibatches per NN training = 1000, Mb = 50, noise = 0.7, we got (60,11) eps 17 w/o box penalty
# minibatches per NN training = 250, Mb = 50, noise = 0.35, we got (63,26) eps 79 w/o box penalty
# minibatches per NN training = 100, Mb = 100, noise = 0.5, we got (61,28) eps 29 w/o box penalty
# minibatches per NN training = 250, Mb = 50, noise = 0.35, we got (62,15) eps 78 w/o box penalty modified quantile to 10% instead of 25%
# minibatches per NN training = 250, Mb = 50, noise = 0.5, we got (61,17) eps 24 w/o box penalty modified quantile to 50% instead of 25%

