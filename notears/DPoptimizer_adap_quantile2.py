import torch
import epsilon_calculation
import utils as ut
import math
import ipdb


def make_optimizer_class(cls):

    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.nbmb = 0
            self.nbclip = 0

            self.gamma = 0.25 #0.1 better ?
            self.b = 0.0
            self.noise_multiplier_b = minibatch_size / microbatch_size / 20.0
            self.noise_multiplier_delta = ( noise_multiplier**(-2) - (2*self.noise_multiplier_b)**(-2) )**(-0.5)
            print("noise_multiplier ",noise_multiplier," noise_multiplier_b ",self.noise_multiplier_b," noise_multiplier_delta ",self.noise_multiplier_delta)
            self.clip = 10.0 #initial clipping threshold
            self.etha = 0.8 #0.4 originally
            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
                #for param in group['params']:
                    #print(param)
            #print(self.param_groups)

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            r = torch.rand(1).item()
            total_norm = 0
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        w = param.grad.clone()
                        total_norm += ( w.norm(2).item() )**2
            total_norm = math.sqrt( total_norm )
            if r < 0.001:
                print("total norm for a microbatch",total_norm,"clip",self.clip)
            clip_coef = min( self.clip / (total_norm + 1e-12) , 1.0 )
            self.nbmb += 1
            if total_norm + 1e-12 > self.clip :
                self.nbclip += 1
            else:
                self.b += 1
            for group in self.param_groups:
                for param, accum_grad in zip( group['params'], group['accum_grads']):
                    if param.requires_grad:
                        w = param.grad.clone()
                        cw = w * clip_coef
                        #if r < 0.001:
                            #print( "grad norm for a microbatch per group", w.norm(2).item(), " clipped grad norm for a microbatch per group", cw.norm(2).item() )
                        accum_grad.add_(cw)
                        #ipdb.set_trace()

        def zero_grad(self):
            super(DPOptimizerClass, self).zero_grad()
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            r = torch.rand(1).item()
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(self.clip * self.noise_multiplier_delta * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)
#calculation of b and update of clip
            self.b += self.noise_multiplier_b * torch.rand(1).item()
            self.b *= self.microbatch_size / self.minibatch_size
            self.clip *= math.exp(-self.etha*(self.b-self.gamma))
            self.b = 0

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
    mb = 1
    noisemult = 1.0
    delta = 1e-5

    if not DPflag:
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.5, momentum=0.9)
    else:
        DPSGD = make_optimizer_class(torch.optim.SGD)
        optimizer = DPSGD(params=linear.parameters(), noise_multiplier=noisemult, minibatch_size=Mb, microbatch_size=mb,
            lr=0.5, momentum=0.9)

    with torch.no_grad():
        X = next(iter(train_data))[0]
        Yhat = linear (X)
        Y = next(iter(train_data))[1]
        globalloss = criterion(Yhat,Y)
        print( "loss on the whole dataset" , globalloss.item() )

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
            #i += 1

        print("Proportion of microbatches that were clipped up to now since start", optimizer.nbclip / optimizer.nbmb )


    iterations = (n/Mb)*numepochs
    print("Total number of minibatches",iterations)

    print(list(linear.parameters()))

    if DPflag:
        print( 'Achieves ({}, {})-DP'.format( epsilon_calculation.epsilon(n, Mb, noisemult, iterations, delta) , delta ) )

    #print("learnt weights of affine transformation \n", list(linear.parameters()) )
    #print("w_true ^ T \n",w_true.t())

if __name__ == '__main__':
    main()

