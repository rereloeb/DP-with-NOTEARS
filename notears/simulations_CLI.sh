

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '8' --boxpenalty 0 --method 'plain_vanilla' \
    > ./outputs/out25

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '3.1 1 6 3' --boxpenalty 0 --method 'grouppping' \
    > ./outputs/out26

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 0 --method 'adaclip' \
    > ./outputs/out27

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 0 --method 'adap_quantile' \
    > ./outputs/out28

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 0 --method 'adaclip_and_adap_quantile' \
    > ./outputs/out29

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 0 --method 'grouppping_and_adap_quantile' \
    > ./outputs/out30


    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '10' --boxpenalty 1 --method 'plain_vanilla' \
    > ./outputs/out31

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '10 3 10.1 3.1 5 3.2' --boxpenalty 1 --method 'grouppping' \
    > ./outputs/out32

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 1 --method 'adaclip' \
    > ./outputs/out33

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 1 --method 'adap_quantile' \
    > ./outputs/out34

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 1 --method 'adaclip_and_adap_quantile' \
    > ./outputs/out35

    python nonlinear.py --samples 5000 --nodes 10 --edges 2 --graphtype 'RE' --SEMtype 'mim' \
    --minibatch 50 --noisemult 0.6 --minibatchesperNNtraining 250 --clip '0' --boxpenalty 1 --method 'grouppping_and_adap_quantile' \
    > ./outputs/out36


