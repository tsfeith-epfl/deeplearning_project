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
"""
class Module(object):
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []
"""

# ACTIVATION FUNCTIONS

class ReLU():
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

    def params():
        return []
    
class Sigmoid():
    def forward(self, input_):
        """
        Sigmoid(x) = 1/(1+e^(-x))
        """
        return 1 / (1 + (-input_).exp())
    
    def backward(self, gradwrtoutput):
        """
        Derivative of sigmoid: dsig(x)/dx = sig(x)(1-sig(x))
        """
        return 1 / (1 + gradwrtoutput.exp()) * (1 - 1 / (1 + gradwrtoutput.exp()))
    
    def params():
        return []
       
## LOSS FUNCTIONS

class MSE():
    def __init__(self):
        pass
    def forward(self, predictions, targets):
        """
        Mean Squared Error: MSE(x) = 1/N * (y - f(x))^2
        """
        self.size = len(predictions)
        self.predictions, self.targets = predictions, targets
        return ((self.targets - self.predictions)**2).sum()/self.size
        
    def backward(self):
        """
        Derivative of MSE = -2/N * (y - f(x))
        """
        self.predictions.grad = 2/self.size * (self.predictions - self.targets)
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
        for p in self.params: p -= lr*p.grad
        
    def zero_grad(self):
        """
        Zero all the gradients.
        """
        for p in self.params: p.grad = 0


class Sequential(): #I may need also functions 
    
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
        output = input_.clone()
        for module in self.model:
            output = module.forward(output)    
        
        return output
    
    def backward(self, gradwrtoutput):
        """
        Do the backward pass of each module and keep track of the gradient
        """
        grad = gradwrtoutput
        for module in self.model[::-1]:
            grad += module.backward(grad)
            
        return grad

    def params(self):
        """
        Gather the new parameters
        """
        params = []
        for module in self.model:
            for param in module.params():
                params.append(param)
            
        return params
    
    def modules(self):
        return self.model


# +
from torch.nn.functional import fold, unfold
##ADD weights initialization

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias = True):
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
            raise Exception("Please enter dilation parameters as tuple or int") 
        
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

        self.use_bias = bias
        self.weight = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).normal_()
        self.weight.grad = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).zero_()
        
        if self.use_bias:
            self.bias = torch.empty(self.in_channels, self.out_channels, 1).zero_()
            self.bias.grad = torch.empty(self.in_channels, self.out_channels, 1).zero_()  
            
    def forward(self, input_):
        """
        Perform convolution as a linear transformation
        """
        self.input = input_
        self.output_shape = (math.floor((input_.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                             math.floor((input_.shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
        output = torch.empty(self.input.shape)
        unfolded = unfold(input_, kernel_size = self.kernel_size,  dilation=self.dilation, padding=self.padding, stride=self.stride)
        
        print(
        
        if self.use_bias:
            wxb = self.weight.view(self.out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)
        else:
            wxb = self.weight.view(self.out_channels, -1) @ unfolded

        actual = wxb.view(input_.shape[0], self.out_channels, self.output_shape[0], self.output_shape[1])
        return actual
        
    def backward(self, grad):
        """
        Compute gradients wrt parameters(w, b) and input(x)
        dL/dw = conv(x, dL/dy)
        dL/db = eye(y.shape)
        dL/dx = 
        """
        
        # compute the gradient of dLdW
        grad_shape = (grad.shape[-2], grad.shape[-1])
        
        unfolded = unfold(self.input.view(self.in_channels, 1, self.input.shape[2], self.input.shape[3]), kernel_size = self.output_shape, dilation = self.stride)
        
        dLdW = (grad.view(self.out_channels,-1) @ unfolded).view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.weight.grad += dLdW
        
        # compute the gradient dLdX
        kernel_mirrored = self.weight.flip([2,3])
        
        expanded_grad = torch.empty(self.input.shape[0], self.out_channels, (grad_shape[0]-1) * (self.stride[0] - 1) + grad_shape[1], (grad_shape[1]-1) * (self.stride[1] - 1) + grad_shape[1]).yero_()
        expanded_grad[:, :, ::self.stride[0], ::self.stride[1]] = grad
        
        unfolded = unfold(expanded_grad, kernel_size = self.kernel_size, padding = (self.kernel_size[0] - 1, self.kernel_size[1] - 1))
        
        corrected_kernel = self.kernel_mirrored.view(self.in_channels, self.kernel_size[0] * self.kernel_size[1] * self.out_channels)
        dLdX = (corrected_kernel @ unfolded).view(self.input.shape)
        
        return dLdX
        
        
        
        """
        self.grad = grad
        unfolded = unfold(self.input, kernel_size = self.kernel_size)#,  dilation=self.dilation
                         # , padding=self.padding, stride=self.stride)
        wxb = grad.view(self.out_channels, -1) @ unfolded
        actual = wxb.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.weight.grad.add_(actual.mean(dim = 0))
        

        # dL/db (I think this is correct, right?)
        self.b.grad.add_(torch.empty(self.bias.shape)).zero_() + 1

        # return dL/dx

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
        self.input.grad = lhs @ unfolded

        self.input.grad = self.input.grad.view(self.input.shape)
        print(self.input.grad.size())

        return self.input.grad
        """
    def params():
        return [self.weight, self.bias]


# -

class Upsampling():
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
        self.conv = Conv2d(self.in_channels, self.out_channels, kernel_size=kernel_size, stride = self.scale_factor)
        nearest_upsampling = NearestUpsampling(self.scale_factor)
        return conv.forward(nearest_upsampling.forward(input_))
    
    def backward(self, grad):
        # STILL NEED TO DO
        pass
    
    def params():
        return self.conv.params()


class NearestUpsampling(Upsampling):
    def __init__(self, scale_factor):
        """
        Store the attributes
        """
        super().__init__(scale_factor)

    def params():
        return []
        
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

class Model():
    def __init__(self):
        """
        Instantiate model, optimizer, loss function, any other stuff needed.

        Returns
        -------
        None
        """
        
        self.model = Sequential(Conv2d(in_channels=3, out_channels=48, kernel_size=3, stride=2),
                                ReLU,
                                Conv2d(in_channels=48, out_channels=48, kernel_size=3, stride=2),
                                ReLU,
                                Upsampling(scale_factor=2),
                                ReLU,
                                Upsampling(scale_factor=2),
                                Sigmoid)
        self.optimizer = SGD(self.model.params, 1e-3)
        self.criterion = MSE()
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.mini_batch_size = 625

    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel.pth into the model.

        Returns
        -------
        None
        """
        from pathlib import Path
        model_path = Path(__file__).parent / "bestmodel.pth"
        model = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(model)
        self.model = self.model.to(self.device)

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
        return self.model.forward(test_input)
