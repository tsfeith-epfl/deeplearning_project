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
        copy = input_.detach().clone()
        return copy.apply_(lambda x: 1/(1+math.exp(-x)))
    
    def backward(self, gradwrtoutput):
        """
        Derivative of sigmoid: dsig(x)/dx = sig(x)(1-sig(x))
        """
        copy = gradwrtoutput.detach().clone()
        return copy.apply_(lambda x: forward(x)*(1-forward(x)))
    
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
        self.w = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).zero_() +1
        self.grad_w = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).zero_()
        
        if self.bias:
            self.b = torch.empty(self.in_channels, self.out_channels, 1).zero_()
            self.grad_b = torch.empty(self.in_channels, self.out_channels, 1).zero_()  

    def forward(self, input_):
        """
        Perform convolution as a linear transformation
        """
        self.input = input_
        output = torch.empty(self.input.shape)
        unfolded = unfold(input_, kernel_size = self.kernel_size,  dilation=self.dilation, padding=self.padding, stride=self.stride)
        if self.bias:
            wxb = self.w.view(self.out_channels, -1) @ unfolded + self.b.view(1, -1, 1)
        else:
            wxb = self.w.view(self.out_channels, -1) @ unfolded

        actual = wxb.view(input_.shape[0], self.out_channels,
                          math.floor((input_.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                          math.floor((input_.shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
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
        wxb = grad.view(self.out_channels, -1) @ unfolded + self.b.view(1, -1, 1)
        actual = wxb.view(1, self.out_channels,
                          math.floor((input_.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                          math.floor((input_.shape[3] + 2*self.padding[1] - self.dialtion[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
        self.grad_w.add_(actual)
        
        ###return dy/dx


# -

# We don't need this, it's transposed convolution OR nearest neighbor, right?
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
        self.input = input_ #same as above
        self.in_channels = input_.shape[1] #[1] if we have more images
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
