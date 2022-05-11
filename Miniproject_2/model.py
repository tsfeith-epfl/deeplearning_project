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


# +
# Suggested structure
# HOWEVER GRADING WILL REWARD ORIGINALITY
class Module(object):
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
    
## ACTIVATION FUNCTIONS
    
class ReLU(Module): 
    def forward(self, input_):
        """
        ReLU(x) = max(0, x): returns the max between 0 and the input
        """
        return max(0, input_)
    
    def backward(self, input_):
        """
        Derivative of ReLU: 1 if input > 0, 0 elsewhere
        """
        return int(input_>0)
    
class Sigmoid(Module):
    def forward(self, input_):
        """
        Sigmoid(x) = 1/(1+e^(-x))
        """
        return 1/(1+math.exp(-input_))
    
    def backward(self, input_):
        """
        Derivative of sigmoid: dsig(x)/dx = sig(x)(1-sig(x))
        """
        return forward(input_)*(1-forward(input_))
    
## LOSS FUNCTIONS

class MSE(Module):
    def __init__(self, predictions, targets):
        self.size = len(predictions)
        self.predictions, self.targets = predictions, targets
    
    def forward(self):
        """
        Mean Squared Error: MSE(x) = 1/N * (y - f(x))^2
        """
        return sum([(self.targets[i] - self.predictions[i])**2 for i in range(self.size)]) / self.size
    
    def backward(self):
        """
        Derivative of MSE = -2/N * (y - f(x))
        """
        return -2*sum([(self.targets[i] - self.predictions[i]) for i in range(self.size)]) / self.size
        
    


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
        output = self.model[0](input_)
        for i in range(1, len(self.model)):
            output = self.model[i](output)
        return output
    
    def backward(self, der):
        """
        Do the backward pass of each module and keep track of the gradient
        """
        grad = der
        for module in self.model[::-1]:
            grad = module.backward(grad)
            
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

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias = False):
        """
        Store the attributes and initialize the parameters and gradient tensors
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dialtion = dilation
        self.w = torch.empty(self.out_channels, self.in_channels, kernel_size[0], kernel_size[1]).zero_()
        self.grad_w = torch.empty(self.out_channels, self.in_channels, kernel_size[0], kernel_size[1]).zero_()
        
        if bias:
            self.b = torch.empty(self.in_channels, out_channels, 1).zero_()
            self.grad_b = torch.empty(self.in_channels, out_channels, 1).zero_()  

    def forward(self, input_):
        """
        Perform convolution as a linear transformation
        """
        self.input = input_
        unfolded = unfold(input_, kernel_size = self.kernel_size,  dilation=self.dialtion
                          , padding=self.padding, stride=self.stride)
        wxb = self.w.view(self.out_channels, -1) @ unfolded + self.b.view(1, -1, 1)
        actual = wxb.view(1, out_channels,
                          math.floor((input_.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                          math.floor((input_.shape[3] + 2*self.padding[1] - self.dialtion[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
        return actual
        
    def backward(self, grad):
        """
        Compute gradients wrt parameters(w, b) and input(x)
        dy/dw = conv(x, dL/dy)
        dy/db = 
        dy/dx = 
        """
        #is grad already a k[0] x k[1] tensor????
        unfolded = unfold(self.input, kernel_size = self.kernel_size,  dilation=self.dialtion
                          , padding=self.padding, stride=self.stride)
        wxb = grad.view(out_channels, -1) @ unfolded + self.b.view(1, -1, 1)
        actual = wxb.view(1, self.out_channels, self.input.shape[2] - self.kernel_size[0] + 1,
                          self.input.shape [3] - self.kernel_size[1] + 1)
        self.grad_w.add_(actual)


# -

# Working space below

import torch
import numpy as np
x = torch.arange(48, dtype=float).view(1,3,4,4)
x

(y.view(4,-1) @ torch.nn.functional.unfold(x, kernel_size = (2,2), stride = 2)).shape

y = torch.arange(48, dtype=float).view(4,3,2,2)
y.view(4,-1).shape

unfolded = torch.nn.functional.unfold(x, kernel_size = (2,2))
actual = y.view(4,-1).float() @ unfolded.float()
actual.view(1, 4, x.shape[2] - y.shape[2] + 1,
                          x.shape[3] - y.shape[3] + 1).shape

y.view(4,-1)

x.shape

# +
in_channels = 3
out_channels = 4
kernel_size = (2 , 2)

# conv = torch . nn . Conv2d ( in˙channels , out˙channels , kernel˙size )
x = torch.randn((1 , in_channels , 32 , 32))
y = torch.arange(48, dtype=float).view(4,3,2,2)
stride = (2,2)

# Output of PyTorch convolution
# Output of convolution as a matrix product
unfolded1 = torch.nn.functional.unfold(x , kernel_size = kernel_size, stride=2)
wxb = y.view(out_channels , -1).float() @ unfolded1.float() #+ conv.bias.view(1 , -1 , 1)
actual = wxb.view(1 , out_channels , math.floor((x.shape[2] - kernel_size[0])/stride[0]) + 1 , math.floor((x.shape[3] - kernel_size[1])/stride[1]) + 1)
actual.shape
# -

math.floor(8.2)


# We need to define the following modules: Conv2d, TransposeConv2d or
# NearestUpsampling, ReLU, Sigmoid, MSE, SGD, Sequential

class Model ():
    def __init__(self):
        """
        Instantiate model, optimizer, loss function, any other stuff needed.

        Returns
        -------
        None
        """
        pass

    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel .pth into the model.

        Returns
        -------
        None
        """
        pass

    def train(self, train_input, train_target):
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

        pass

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

        pass
