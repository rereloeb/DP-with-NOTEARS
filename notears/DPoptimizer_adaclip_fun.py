import torch
import epsilon_calculation
import utils as ut
import math
import ipdb


def make_optimizer_class(cls):

    class DPOptimizerClass(cls):
        def __init__(self, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):

            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size
            self.nbmb = 0
            self.nbclip = 0

#adaclip params
            self.h1 = 1e-6
            #self.h2 = 1e+1
            self.beta1 = 0.9
            self.beta2 = 0.9
            self.gamma = 1.0 #target for ||w||^2
            self.clip = 1.0 #initial clip

#adap_quantile params
            self.gamma2 = 0.5 #target quantile for the clip
            self.b = 0.0
            self.noise_multiplier_b = minibatch_size / microbatch_size / 20.0
            self.noise_multiplier_delta = ( noise_multiplier**(-2) - (2*self.noise_multiplier_b)**(-2) )**(-0.5)
            self.etha = 0.0 #speed of change for the clip

            self.size = 0
            for group in self.param_groups:
                for param in group['params']:
                    if param.requires_grad:
                        self.size += torch.numel(param.data)
            print(self.size)
            self.w1 = 0
            self.w2 = 0
            self.w1_a = 0
            self.w2_a = 0
            

            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
#a vector
                group['a_mat'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]
                #group['s_mat'] = [torch.ones_like(param.data)*math.sqrt(self.h1*self.h2) if param.requires_grad else None for param in group['params']]
                #group['b_mat'] = [torch.ones_like(param.data)*math.sqrt(self.h1*self.h2*self.size/self.gamma) if param.requires_grad else None for param in group['params']]
#variance vector
                group['s_mat'] = [torch.ones_like(param.data) if param.requires_grad else None for param in group['params']]
#variance capped and floored
                group['s2_mat'] = [torch.ones_like(param.data) if param.requires_grad else None for param in group['params']]
#b vector
                group['b_mat'] = [torch.ones_like(param.data)*math.sqrt(self.size/self.gamma) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            r = torch.rand(1).item()

            total_norm_a = 0.0
            total_norm = 0.0
            for group in self.param_groups:
                for param, a, b in zip( group['params'], group['a_mat'], group['b_mat']):
                    if param.requires_grad:
                        total_norm_a += ( param.grad.clone().detach().norm(2).item() )**2
                        w = ( param.grad.clone().detach() - a ) / b
                        total_norm += ( w.norm(2).item() )**2

            self.w2_a = self.beta1 * self.w2_a + (1-self.beta1) * total_norm_a
            self.w2 = self.beta1 * self.w2 + (1-self.beta1) * total_norm
            #if r < 0.001:
                #print("||grad||^2",total_norm_a)
                #print("exp ma of ||grad||^2",self.w2_a)
                #print("||w||^2",total_norm)
                #print("exp ma of ||w||^2",self.w2)

            total_norm_a = math.sqrt( total_norm_a )
            total_norm = math.sqrt( total_norm )

            self.w1_a = self.beta1 * self.w1_a + (1-self.beta1) * total_norm_a
            self.w1 = self.beta1 * self.w1 + (1-self.beta1) * total_norm
            if r < 0.001:
                #print("||grad||",total_norm_a)
                print("exp ma of ||grad||",self.w1_a)            
                #print("||w||",total_norm)
                print("exp ma of ||w||",self.w1)

            clip_coef = min( self.clip / (total_norm + 1e-12) , 1.0 )

            self.nbmb += 1
            if total_norm + 1e-12 > self.clip :
                self.nbclip += 1
            else:
                self.b += 1

            for group in self.param_groups:
                for param, accum_grad, a, b in zip( group['params'], group['accum_grads'], group['a_mat'], group['b_mat'] ):
                    if param.requires_grad:
                        w = ( param.grad.clone().detach() - a ) / b
                        cw = w #* clip_coef
                        nw = cw + self.noise_multiplier_delta * self.clip * torch.randn_like(param.grad.data) * math.sqrt(self.microbatch_size / self.minibatch_size)
                        w2 = nw * b + a
                        accum_grad.add_(w2)
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

#update the clip
            self.b += self.noise_multiplier_b * torch.rand(1).item()
            self.b *= self.microbatch_size / self.minibatch_size
            self.clip *= math.exp(-self.etha*(self.b-self.gamma2))
            self.b = 0

#update of a_mat (the mean estimation), s_mat (the var estimation), b_mat (the stdev estimation)
            r = torch.rand(1).item()
            for group in self.param_groups:
                for param, accum_grad, a, b, s, s2 in zip(group['params'], group['accum_grads'], group['a_mat'], group['b_mat'], group['s_mat'], group['s2_mat']):
                    if param.requires_grad:
#update a
                        a.mul_(self.beta1)
                        a.add_( param.grad.clone().detach() * (1-self.beta1) )
#calc v and update s, s2
                        v =  ( (param.grad.clone().detach() - a)**2 - (b**2) * ((self.noise_multiplier_delta*self.clip)**2) * ((self.microbatch_size / self.minibatch_size)**2)) * self.minibatch_size / self.microbatch_size
                        if r < 0.001:
                            print("v", v)
                        s.copy_( s * self.beta2 + v * (1.0-self.beta2) )
                        s2.copy_( torch.max(s , torch.ones_like(param.grad.data)*self.h1 ) )
#update b
            total_norm1 = 0.0
            for group in self.param_groups:
                for param, s2 in zip( group['params'], group['s2_mat'] ):
                    if param.requires_grad:
                        total_norm1 += torch.sqrt(s2).norm(1).item()
            for group in self.param_groups:
                for param, a, b, s , s2 in zip( group['params'], group['a_mat'], group['b_mat'], group['s_mat'], group['s2_mat'] ):
                    if param.requires_grad:
#opti proposed in paper
                        b.copy_( torch.sqrt(torch.sqrt(s2)) * math.sqrt(total_norm1/self.gamma) )
#traditional withening of grad
                        #b.copy_( torch.sqrt(s2) * math.sqrt(self.size/self.gamma) )
                        if r < 0.001:
                            print("a after update for 1 param", a)
                            print("s after update for 1 param", s)
                            print("s2 after update for 1 param", s2)
                            print("b after update for 1 param", b)
                            print("clip", self.clip)

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

