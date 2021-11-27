import torch
import epsilon_calculation
import utils as ut
import math
import ipdb


def make_optimizer_class(cls):

    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.nbmb = 0
            self.nbclip = 0

            self.h1 = 1e-12
            self.h2 = 1e+1
            self.beta1 = 0.99
            self.beta2 = 0.9
            self.gamma = 1.0
            self.clip = 1.0

            self.size = 0
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        self.size += torch.numel(param.data)
            print(self.size)
            self.w1 = 0
            self.w2 = 0

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
                group['a_mat'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
                group['s_mat'] = [torch.ones_like(param.data)*math.sqrt(self.h1*self.h2) if param.requires_grad else None for param
                    in group['params']]
                group['b_mat'] = [torch.ones_like(param.data)*math.sqrt(self.h1*self.h2*self.size/self.gamma) if
                    param.requires_grad else None for param in group['params']]
                #group['s_mat'] = [torch.ones_like(param.data) if param.requires_grad else None for param
                    #in group['params']]
                #group['b_mat'] = [torch.ones_like(param.data)*math.sqrt(self.size/self.gamma) if
                    #param.requires_grad else None for param in group['params']]

                #for param in group['params']:
                    #print(param)

            #print(self.param_groups)

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            r = torch.rand(1).item()
            total_norm = 0.0
            for group in self.param_groups:
                for param, a, b in zip( group['params'], group['a_mat'], group['b_mat']):
                    if param.requires_grad:
                        w = ( param.grad.clone().detach() - a ) / b
                        total_norm += ( w.norm(2).item() )**2
                        #if r < 0.0001:
                            #print("g",param.grad)
                            #print("w = (g - a)/b",w)

            self.w2 = self.beta1 * self.w2 + (1-self.beta1) * total_norm
            if r < 0.001:
                print("||w||^2",total_norm)
                print("exp ma of ||w||^2",self.w2)

            total_norm = math.sqrt( total_norm )

            self.w1 = self.beta1 * self.w1 + (1-self.beta1) * total_norm
            if r < 0.001:
                print("||w||",total_norm)
                print("exp ma of ||w||",self.w1)

            clip_coef = min( self.clip / (total_norm + 1e-12) , 1.0 )

            self.nbmb += 1
            if total_norm + 1e-12 > self.clip :
                self.nbclip += 1

            for group in self.param_groups:
                for param, accum_grad, a, b in zip( group['params'], group['accum_grads'], group['a_mat'], group['b_mat'] ):
                    if param.requires_grad:
                        w = ( param.grad.clone().detach() - a ) / b
                        cw = w * clip_coef
                        nw = cw + self.noise_multiplier * self.clip * torch.randn_like(param.grad.data) * math.sqrt(self.microbatch_size / self.minibatch_size)
                        w2 = nw * b + a
                        #if r < 0.001:
                            #print(1.0/clip_coef)
                            #print( "grad ", param.grad.norm(2).item(), " a ", a.norm(2).item(), " b ", b.norm(2).item(),
                                #" whitened grad ", w.norm(2).item(), " clipped grad ", cw.norm(2).item(), " clipped + noised grad ",
                                #nw.norm(2).item(), " rescaled grad ",w2.norm(2).item() )
                            #print( "clipped norm for a microbatch per group", cw.norm(2).item() )
                        accum_grad.add_(w2)
                        #print(param, param.grad.data, a, b, accum_grad)
                        #ipdb.set_trace()

        def zero_grad(self):
            super(DPOptimizerClass, self).zero_grad()
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad:
                        param.grad.data = accum_grad.clone()
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)

            super(DPOptimizerClass, self).step(*args, **kwargs)

#update of a_mat (the mean estimation), s_mat (the var estimation), b_mat (the stdev estimation)
            r = torch.rand(1).item()
            for group in self.param_groups:
                for param, accum_grad, a, b, s in zip(group['params'], group['accum_grads'], group['a_mat'], group['b_mat'], 
                    group['s_mat']):
                    if param.requires_grad:
#update a
                        a.mul_(self.beta1)
                        a.add_( param.grad.clone().detach() * (1-self.beta1) )
#calc v (with cap and floor) and update s
                        y =  ( (param.grad.clone().detach() - a)**2 - (b**2) * ((self.noise_multiplier*self.clip)**2) * ((self.microbatch_size / self.minibatch_size)**2)
                            ) * self.minibatch_size / self.microbatch_size
                        z = torch.max(y , torch.ones_like(param.grad.data)*self.h1 )
                        v = torch.min( z , torch.ones_like(param.grad.data)*self.h2)
                        if r < 0.001:
                            print("v before min max", y)
                            print("v", v)
                        s.copy_( torch.sqrt( torch.square(s) * self.beta2 + v * (1.0-self.beta2) ) )
#update b
            total_norm1 = 0.0
            for group in self.param_groups:
                for param, s in zip( group['params'], group['s_mat'] ):
                    if param.requires_grad:
                        total_norm1 += s.norm(1).item()
            for group in self.param_groups:
                for param, a, b, s in zip( group['params'], group['a_mat'], group['b_mat'], group['s_mat'] ):
                    if param.requires_grad:
#opti proposed in paper
                        b.copy_( torch.sqrt(s) * math.sqrt(total_norm1/self.gamma) )
#traditional withening of grad
                        #b.copy_( s * math.sqrt(self.size/self.gamma) )
                        if r < 0.001:
                            print("a after update for 1 param", a)
                            print("s after update for 1 param", s)
                            print("b after update for 1 param", b)

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

        print("Proportion of microbatches that were clipped up to now since start", optimizer.nbclip / optimizer.nbmb )


    iterations = (n/Mb)*numepochs
    print("Total number of minibatches",iterations)

    print(list(linear.parameters()))

    #if DPflag:
        #print( 'Achieves ({}, {})-DP'.format( epsilon_calculation.epsilon(n, Mb, noisemult, iterations, delta) , delta ) )

    #print("learnt weights of affine transformation \n", list(linear.parameters()) )
    #print("w_true ^ T \n",w_true.t())

if __name__ == '__main__':
    main()

