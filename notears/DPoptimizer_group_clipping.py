import torch
import epsilon_calculation
import utils as ut
import numpy as np
import math

def make_optimizer_class(cls):

    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            #print(self.param_groups)
            i=0
            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
#1 line in l2_norm_clip (2d array) per group, to be recoded if models get more complicated (more groups in param_groups)
                group['l2_norm_clips'] = l2_norm_clip[i]
                i += 1
                group['nbmb'] = [torch.zeros(1) if param.requires_grad else None for param in group['params']]
                group['nbclip'] = [torch.zeros(1) if param.requires_grad else None for param in group['params']]
                #for x in group['params']:
                    #print(x)
            #print(self.param_groups)

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

#for one microbatch, for each param, gradient is clipped and added to the accum_grads of the param
#(accum_grads is an extra attribute of the optimizer we have defined above)
        def microbatch_step(self):
            r = torch.rand(1).item()
            if r < 0.001:
                print("Grad norm per param group for one microbatch")
            for group in self.param_groups:
                for param, accum_grad, clip, n1, n2 in zip( group['params'], group['accum_grads'], group['l2_norm_clips'], group['nbmb'],
                    group['nbclip']):
                    if param.requires_grad:
                        total_norm = param.grad.data.norm(2).item()
                        if r < 0.001:
                            print(total_norm)
                        clip_coef = min( clip / (total_norm + 1e-6) , 1.0 )
                        n1.add_(1)
                        if total_norm + 1e-12 > clip :
                            n2.add_(1)
                        accum_grad.add_(param.grad.data.mul(clip_coef))
                        #print(param, param.grad.data, clip, param.grad.data.mul(clip_coef), accum_grad)

        def zero_grad(self):
            super(DPOptimizerClass, self).zero_grad()
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

#for each minibatch, noise is added to the clipped gradients accumulated over several microbatches (accum_grads)
#and params will be updated in the model
        def step(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad, clip in zip(group['params'], group['accum_grads'], group['l2_norm_clips'] ):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_( clip * self.noise_multiplier * torch.randn_like(param.grad.data) )
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass

def show_params_and_gradients(model):
    for f in model.parameters():
        print('Param is',f.data[0])
        print('Gradient is ',f.grad[0])

def main():

    torch.set_default_dtype(torch.double)
    ut.set_random_seed(55)

    train_data = torch.utils.data.TensorDataset(
    torch.Tensor(100 * [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]),
    torch.Tensor(100 * [
        [0],
        [0],
        [0],
        [1],
        [0],
        [1],
        [1],
        [1],
    ]))

    n = len(train_data)

    DPflag = True

# creating a model = an affine transformation
# learning the weights of the affine transformation that sends input to target
# based on square loss error using various gradient descent algorithms

    linear = torch.nn.Sequential( torch.nn.Linear(len(next(iter(train_data))[0]), len(next(iter(train_data))[1])), torch.nn.Sigmoid() )
    criterion = torch.nn.BCELoss()

# minibatch size
    numepochs = 20
    Mb= 20
# for DP: microbatch size, noise multiplier, delta, clip
    mb = 10
    noisemult = 1.0
    delta = 1e-5
#each line in clip refers to a group (pytorch sense) of params in the optimizer
#each element in a given line refers to a param in the group (requires_grad=True)
    clip = [ [ 0.015 , 0.01 ] ]

    if not DPflag:
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.5, momentum=0.9)
    else:
        DPSGD = make_optimizer_class(torch.optim.SGD)
        G = len(np.unique(clip))
        noisemult_modified = noisemult * math.sqrt(G)
        print("number of groups for clipping ",G," noise multiplier modified for group clipping ",noisemult_modified)
        optimizer = DPSGD(params=linear.parameters(), l2_norm_clip=clip, noise_multiplier=noisemult_modified, minibatch_size=Mb, microbatch_size=mb,
            lr=0.5, momentum=0.9)

    #X = next(iter(train_data))[0]
    #Yhat = linear (X)
    #Y = next(iter(train_data))[1]
    #globalloss = criterion(Yhat,Y)
    #print( "loss on the whole dataset" , globalloss.item() )

    for i in range(numepochs):
        train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = Mb, shuffle = True)
        #print("epoch ",i)
        j = 0
        for x_batch, y_batch in train_loader:
            #print("minibatch",j)
            if not DPflag:
                optimizer.zero_grad()
                output = linear(x_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                train_data2 = torch.utils.data.TensorDataset(x_batch, y_batch)
                train_loader2 = torch.utils.data.DataLoader(dataset = train_data2, batch_size = mb, shuffle = True)
                k = 0
                for X_microbatch, y_microbatch in train_loader2:
                    #print("microbatch",k)
                    optimizer.zero_microbatch_grad()
                    loss = criterion(linear(X_microbatch), y_microbatch)
                    loss.backward()
                    #show_params_and_gradients(linear)
                    optimizer.microbatch_step()
                    k += 1
                #print("before optimizer step")
                #show_params_and_gradients(linear)
                optimizer.step()
                #print("after optimizer step")
                #show_params_and_gradients(linear)
            j += 1

        with torch.no_grad():
            X = next(iter(train_data))[0]
            Yhat = linear (X)
            Y = next(iter(train_data))[1]
            globalloss = criterion(Yhat,Y)
            print( "Epoch ",i,"loss on the whole dataset" , globalloss.item() )
            i += 1

        for group in optimizer.param_groups:
            prop = [ x.item() / y.item() for x, y in zip(group['nbclip'], group['nbmb']) ]
            print("Proportion of microbatches that were clipped up to now since start ", prop)


    iterations = (n/Mb)*numepochs
    print("Total number of minibatches",iterations)

    #print(list(linear.parameters()))

    if DPflag:
        print( 'Achieves ({}, {})-DP'.format( epsilon_calculation.epsilon(n, Mb, noisemult, iterations, delta) , delta ) )

    #print("learnt weights of affine transformation \n", list(linear.parameters()) )
    #print("w_true ^ T \n",w_true.t())

if __name__ == '__main__':
    main()

