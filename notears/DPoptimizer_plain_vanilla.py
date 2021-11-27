import torch
import epsilon_calculation

def make_optimizer_class(cls):

    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)
            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.nbmb = 0
            self.nbclip = 0
            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
            #print(self.param_groups)

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

#for one microbatch, total gradient is clipped and added to the accum_grads (an extra param of the optimizer we have defined above)
        def microbatch_step(self):
            total_norm = 0.
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5
            if torch.rand(1).item() < 0.001:
                print("microbatch grad before clipping",total_norm)
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)
            self.nbmb += 1
            if total_norm + 1e-6 > self.l2_norm_clip :
                self.nbclip += 1
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        accum_grad.add_(param.grad.data.mul(clip_coef))
            #print("optimizer params_groups after microbatch step",self.param_groups)

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
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            super(DPOptimizerClass, self).step(*args, **kwargs)

    return DPOptimizerClass

def main():

    DPSGD = make_optimizer_class(torch.optim.SGD)
    torch.set_default_dtype(torch.double)

    n, d, out, j = 10000, 3000, 10, 0
    input = torch.randn(n, d)
    #print("input \n",input)
    w_true = torch.rand(d, out)
    w_true[j, :] = 0
    #print("w_true \n",w_true)
    target = torch.matmul(input, w_true)
    #print("target = input * w_true \n",target)

    DPflag = True

# creating a model = an affine transformation
# learning the weights of the affine transformation that sends input to target
# based on square loss error using various gradient descent algorithms

    linear = torch.nn.Linear(d, out)
    criterion = torch.nn.MSELoss()
# number of epochs, minibatch size
    numepochs = 20
    Mb= 1000
# for DP: microbatch size, noise multiplier, delta, clip
    mb = 20
    noisemult = 0.1
    delta = 1e-5
    clip = 5.0

    if not DPflag:
        optimizer = torch.optim.SGD(linear.parameters(), lr=0.2, momentum=0.9)
    else:
        optimizer = DPSGD(params=linear.parameters(), l2_norm_clip=clip, noise_multiplier=noisemult, minibatch_size=Mb, microbatch_size=mb, lr=0.2, momentum=0.9)

    train_data = torch.utils.data.TensorDataset(input,target)

    for i in range(numepochs):
        train_loader = torch.utils.data.DataLoader(dataset = train_data, batch_size = Mb, shuffle = True)
        print("epoch ",i)
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
                    optimizer.microbatch_step()
                    k += 1
                optimizer.step()
            j += 1
        print("loss ", loss.item())
        i += 1

    iterations = (n/Mb)*numepochs
    print("Total number of minibatches",iterations)

    if DPflag:
        print( 'Achieves ({}, {})-DP'.format( epsilon_calculation.epsilon(n, Mb, noisemult, iterations, delta) , delta ) )

    #print("learnt weights of affine transformation \n", list(linear.parameters()) )
    #print("w_true ^ T \n",w_true.t())

if __name__ == '__main__':
    main()

