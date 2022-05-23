import math
import torch
torch.set_grad_enabled(False)


# we want to build the following model
# Sequential (Conv(stride 2),
#             ReLU,
#             Conv(stride 2),
#             ReLU,
#             Upsampling,
#             ReLU,
#             Upsampling,
#             Sigmoid)

# Suggested structure
# HOWEVER GRADING WILL REWARD ORIGINALITY

class Module(object):
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []


# # ACTIVATION FUNCTIONS

class ReLU(Module): 
    def forward(self, input_):
        """
        ReLU(x) = max(0, x): returns the max between 0 and the input
        """
        input_[input_<torch.empty(input_.shape).zero_()]=0
        return input_
    
    def backward(self, gradwrtoutput):
        """
        Derivative of ReLU: 1 if input > 0, 0 elsewhere
        """
        zeros = torch.empty(gradwrtoutput.shape).zero_() 
        zeros[gradwrtoutput > zeros] = 1
        return zeros

class Sigmoid(Module):
    def forward(self, input_):
        """
        Sigmoid(x) = 1/(1+e^(-x))
        """
        return 1 / (1 + input_.exp())
    
    def backward(self, gradwrtoutput):
        """
        Derivative of sigmoid: dsig(x)/dx = sig(x)(1-sig(x))
        """
        return 1 / (1 + gradwrtoutput.exp()) * (1 - 1 / (1 + gradwrtoutput.exp()))
       
## LOSS FUNCTIONS

class MSE(Module):
    def __init__(self, predictions, targets):
        self.size = len(predictions)
        self.predictions, self.targets = predictions, targets
    
    def forward(self):
        """
        Mean Squared Error: MSE(x) = 1/N * (y - f(x))^2
        """
        return ((self.targets - self.predictions)**2).sum()/self.size
        
    def backward(self):
        """
        Derivative of MSE = -2/N * (y - f(x))
        """
        return -2*(self.targets - self.predictions).sum()/self.size
        
# -

class SGD():
    """
    Stochastic Gradient Descent optimizer
    """
    def __init__(self, params, lr):
        """
        Inputs: model's parameters and learning rate
        """
        self.params = params
        self.lr = lr
    
    def step(self):
        """
        Perform one step of Stochastig Gradient Descnet
        """
        for p, grad in self.params: p -= lr*grad
        
    def zero_grad(self):
        """
        Zero all the gradients.
        """
        for p, grad in self.params: grad = 0


class Sequential(Module): #I may need also functions 
    
    def __init__(self, *args):
        """
        Initialize an empty list in which we're going to append the modules
        """
        self.model = []
        for module in args:
            self.model.append(module)
            
    def forward(self, input_):
        """
        Do the forward pass of each module and keep track of the output
        """
        output = self.model[0].forward(input_)
        for i in range(1, len(self.model)):
            output = self.model[i].forward(output)
        return output
    
    def backward(self, gradwrtoutput):
        """
        Do the backward pass of each module and keep track of the gradient
        """
        grad = gradwrtoutput
        for module in self.model[::-1]:
            grad += module.backward(grad)
            
        return grad

    def param(self):
        """
        Gather the new parameters
        """
        params = []
        for module in self.model:
            params.append(module.param())
            
        return params


# +
from torch.nn.functional import fold, unfold
##ADD weights initialization

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias = False):
        """
        Store the attributes and initialize the parameters and gradient tensors
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size,kernel_size)
        elif isinstance(kernel_size, tuple):
            self.kernel_size = kernel_size
        else: 
            raise Exception("Please enter kernel size parameters as tuple or int")
                
        if isinstance(stride, int):
            self.stride = (stride,stride)
        elif isinstance(stride, tuple):
            self.stride = stride
        else: 
            raise Exception("Please enter stride parameters as tuple or int")
            
        if isinstance(dilation, int):
            self.dilation = (dilation,dilation)
        elif isinstance(dilation, tuple):
            self.dilation = dilation
        else: 
            raise Exception("Please enter dialtion parameters as tuple or int") 
        
        if isinstance(padding, int):
            self.padding = (padding,padding)
        elif isinstance(padding, tuple):
            self.padding = padding
        elif padding == 'same':
            # OUT = [(IN−D(K-1)+2P-1)/S]+1
            # SO, FOR OUT = IN WE NEED
            # P = [S(IN-1)-IN+D(K-1)+1]/2
            pad0 = (self.stride[0] * (self.in_channels - 1) - self.in_channels + self.dilation[0] * (self.kernel[0] - 1))//2
            pad1 = (self.stride[1] * (self.in_channels - 1) - self.in_channels + self.dilation[1] * (self.kernel[1] - 1))//2
            self.padding = (pad0, pad1)
        elif padding == 'valid':
            self.padding = (0, 0)
        else:
            raise Exception("Please enter padding parameters as tuple or int, or a string in {\"same\", \"valid\"}")

        self.bias = bias
        self.w = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).normal_()
        self.grad_w = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).zero_()
        
        if self.bias:
            self.b = torch.empty(self.in_channels, self.out_channels, 1).zero_()
            self.grad_b = torch.empty(self.in_channels, self.out_channels, 1).zero_()  

    def forward(self, input_):
        """
        Perform convolution as a linear transformation
        """
        self.input = input_
        self.output_shape = (math.floor((input_.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                             math.floor((input_.shape[3] + 2*self.padding[1] - self.dialtion[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
        output = torch.empty(self.input.shape)
        unfolded = unfold(input_, kernel_size = self.kernel_size,  dilation=self.dilation, padding=self.padding, stride=self.stride)
        if self.bias:
            wxb = self.w.view(self.out_channels, -1).double() @ unfolded + self.b.view(1, -1, 1) #added .double
        else:
            wxb = self.w.view(self.out_channels, -1).double() @ unfolded

        actual = wxb.view(input_.shape[0], self.out_channels, self.output_shape[0], self.output_shape[1])
        return actual
        
    def backward(self, grad):
        """
        Compute gradients wrt parameters(w, b) and input(x)
        dL/dw = conv(x, dL/dy)
        dL/db = eye(y.shape)
        dL/dx = 
        """
        # still not working
        self.grad = grad
        unfolded = unfold(self.input, kernel_size = self.kernel_size)#,  dilation=self.dilation
                         # , padding=self.padding, stride=self.stride)
        wxb = grad.view(self.out_channels, -1) @ unfolded
        actual = wxb.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.grad_w.add_(actual.mean(dim = 0))
        

        # dL/db (I think this is correct, right?)
        self.grad_b.add_(torch.empty(self.b.shape)).zero_() + 1

        ###return dL/dx
        # dL/ds = grad
        # dL/dx = dL/ds * ds/dx

        # First attempt, with dilation = 1.

        #mirror_kernel = self.weight.flip([2,3])

        #zeros = torch.empty(self.input.shape[0], self.out_channels, ).zero_()

        #grad_unfold = unfold(grad, (self.kernel_size[1], self.kernel_size[0])

        #grad_unfold = unfold(grad, (self.kernel_size[1], self.kernel_size[0]))
        #grad_unfold = grad_unfold.transpose(1,2).matmul(mirror_kernel.view(mirror_kernel.shape[0], -1).t()).transpose(1,2)
        #grad_unfold = grad_unfold.view(self.input.shape)

        # Convolution is equivalent with Unfold + Matrix Multiplication + Fold (or view to output shape)
        #inp = torch.randn(1, 3, 10, 12)
        #w = torch.randn(2, 3, 4, 5)
        #inp_unf = torch.nn.functional.unfold(inp, (4, 5))
        #out_unf = inp_unf.transpose(1, 2).matmul(w.view(w.size(0), -1).t()).transpose(1, 2)
        #out = torch.nn.functional.fold(out_unf, (7, 8), (1, 1))
        # or equivalently (and avoiding a copy),
        # out = out_unf.view(1, 2, 7, 8)        


        # unstride the output gradient

        # second index of zeros is output channels
        x, y = output_gradient.size()[-2:]
        zeros = torch.empty(self.input.size(0),self.out_channel, (x-1)*(self.stride-1)+x, y+(y-1)*(self.stride-1)).zero_()
        zeros[:,:,::self.stride,::self.stride] = output_gradient

        self.unstrided_gradient = zeros
        print('self.unstrided_gradient.size()', self.unstrided_gradient.size())


        unfolded = unfold(self.unstrided_gradient,
                          kernel_size = (self.kernel_size,self.kernel_size),
                          stride = 1,
                          padding = (self.kernel_size[0] - 1, self.kernel_size[1] - 1))
        print(unfolded)
        print('unfolded.size()', unfolded.size())

        lhs = self.kernel_flipped.view(self.in_channel, self.kernel_size[0] * self.kernel_size[1] * self.out_channel)
        print('lhs.size()', lhs.size())
        self.input_grad = lhs @ unfolded

        self.input_grad = self.input_grad.view(self.input.shape)
        print(self.input_grad.size())

        return self.input_grad



        return dLdx


# -

class Upsampling(Module):
    def __init__(self, scale_factor):
        """
        Store the attributes
        """
        self.scale_factor = scale_factor

    def forward(self, input_):
        """
        perform upsampling using nearest neighbor rule and then convolution, to have a transposed convolution
        """
        self.input = input_ 
        self.in_channels = input_.shape[1]
        self.out_channels = self.in_channels
        kernel_size = (self.scale_factor, self.scale_factor)
        conv = Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride = self.scale_factor)
        nearest_upsampling = NearestUpsampling(self.scale_factor)
        return conv.forward(nearest_upsampling.forward(input_))
    
    def backward(self, grad):
        pass


class NearestUpsampling(Upsampling):
    def __init__(self, scale_factor):
        """
        Store the attributes
        """
        super().__init__(scale_factor)

    def forward(self, input_):
        """
        Perform upsampling using nearest neighbor rule
        """
        return input_.repeat_interleave(self.scale_factor,3).repeat_interleave(self.scale_factor,2)
    
    def backward(self, grad):
        """
        Convolve the gradient with a filter of ones to return the correct value
        """
        self.filter_ones = torch.empty(self.scale_factor**2, dtype = float).zero_() + 1
        unfolded = unfold(grad, kernel_size = self.scale_factor,
                          stride=self.scale_factor).view(grad.shape[0], grad.shape[1], self.scale_factor*self.scale_factor,
                                                         grad.shape[2]//self.scale_factor*grad.shape[3]//self.scale_factor)
        wxb = self.filter_ones@unfolded
        actual = wxb.view(grad.shape[0], grad.shape[1], grad.shape[2]//self.scale_factor,grad.shape[3]//self.scale_factor)
        return actual

# Working space below

import torch
import numpy as np
x = torch.rand(2,3,4,4)
x

class Model ():
    def __init__(self):
        """
        Instantiate model, optimizer, loss function, any other stuff needed.

        Returns
        -------
        None
        """
        
        self.model = Sequential(Conv2d(stride=2),
                                ReLU,
                                Conv2d(stride=2),
                                ReLU,
                                Upsampling(scale_factor=2),
                                ReLU,
                                Upsampling(scale_factor=2),
                                Sigmoid)
        self.optimizer = SGD(self.model.params, 1e-3)
        self.criterion = MSE()
        
        self.mini_batch_size = 625
        
            
        # STILL NEED TO DO FUNCTION TO INITIALIZE WEIGHTS
        # Initialize weights
        self.model._init_weights()

    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel.pth into the model.

        Returns
        -------
        None
        """
        best_state_dict = torch.load('bestmodel.pth', map_location='cpu')
        self.load_state_dict(best_state_dict)

    def train(self,
              train_input,
              train_target,
              epochs):
        """
        Train the model.

        Parameters
        ----------
        train_input : torch.Tensor
            Tensor of size (N, C, H, W) containing a noisy version of the images
        train_target : torch.Tensor
            Tensor of size (N, C, H, W) containing another noisy version of the
            same images

        Returns
        -------
        None
        """
        print('\nTRAINING STARTING...')
        for e in range(epochs):
            epoch_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                self.optimizer.zero_grad()
                train = train_input.narrow(0, b, self.mini_batch_size)
                target = train_target.narrow(0, b, self.mini_batch_size)
                train, target = data_augmentations(train, target, use_crops = use_crops, n_local_crops = n_local_crops)
                output = self.mode.forward(train, sharpen, sharpen_factor)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()/n_samples
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {e+1}: Loss = {epoch_loss};")

    def predict(self, test_input):
        """
        Perform inference.

        Parameters
        ----------
        test_input : torch.Tensor
            Tensor of size (N1, C, H, W) that has to be denoised by the trained
            or loaded network.

        Returns
        -------
        denoised_signal: torch.Tensor
            Tensor of size (N1, C, H, W) containing the denoised signal.
        """
        return self.forward(test_input)
