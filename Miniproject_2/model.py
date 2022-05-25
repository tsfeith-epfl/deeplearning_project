import math
import torch
torch.set_grad_enabled(True)


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

    def params(self):
        return []
    
class Sigmoid():
    def forward(self, input_):
        """
        Sigmoid(x) = 1/(1+e^(-x))
        """
        return 1 / (1 + (-input_).exp())
    
    def backward(self, grad):
        """
        Derivative of sigmoid: dsig(x)/dx = sig(x)(1-sig(x))
        """
        return 1 / (1 + (-grad).exp()) * (1 - 1 / (1 + (-grad).exp()))
    
    def params(self):
        return []
       
## LOSS FUNCTIONS

class MSE():
    def __init__(self):
        pass
    def forward(self, predictions, targets):
        """
        Mean Squared Error: MSE(x) = 1/N * (y - f(x))^2
        """
        self.size = predictions.numel()
        self.predictions, self.targets = predictions, targets
        return ((self.predictions - self.targets)**2).mean()
        
    def backward(self):
        """
        Derivative of MSE = 2/N * (y - f(x))
        """
        return 2/self.size * (self.predictions - self.targets)

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
        for param in self.params:
            param -= self.lr*param.grad
        
    def zero_grad(self):
        """
        Zero all the gradients.
        """
        for param in self.params: param.grad = torch.empty(param.shape).zero_()


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
            self.bias = torch.empty(self.out_channels).uniform_()
            self.bias.grad = torch.empty(self.out_channels).zero_()  
            
    def forward(self, input_):
        """
        Perform convolution as a linear transformation
        """
        self.input = input_
        self.output_shape = (math.floor((input_.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                             math.floor((input_.shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
        output = torch.empty(self.input.shape)
        unfolded = unfold(input_, kernel_size = self.kernel_size,  dilation=self.dilation, padding=self.padding, stride=self.stride).to(torch.float)
        
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
        
        # compute dLdW
        self.grad = grad.sum(0).unsqueeze(0)
        self.input = self.input.sum(0).unsqueeze(0)
        
        unfolded=unfold(self.input, kernel_size=(self.grad.shape[2], self.grad.shape[3]), dilation=self.stride).view(self.input.shape[0],
                                                                                                                     self.in_channels,
                                                                                                                     self.grad.shape[2] * self.grad.shape[3],
                                                                                                                     self.kernel_size[0] * self.kernel_size[1])

        wxb = self.grad.view(self.out_channels,-1) @ unfolded
        actual = wxb.view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        self.weight.grad.add_(actual.mean(dim = 0))

        # compute the gradient dLdb
        self.bias.grad += self.grad.sum((0,2,3))

        # compute the gradient dLdX
        kernel_mirrored = self.weight.flip([2,3])

        expanded_grad = torch.empty(self.input.shape[0], self.out_channels, (grad.shape[2]-1) * (self.stride[0] - 1) + grad.shape[2], (grad.shape[3]-1) * (self.stride[1] - 1) + grad.shape[3]).zero_()
        expanded_grad[:, :, ::self.stride[0], ::self.stride[1]] = self.grad

        unfolded = unfold(expanded_grad, kernel_size = self.kernel_size, padding = (self.kernel_size[0] - 1, self.kernel_size[1] - 1))

        corrected_kernel = kernel_mirrored.view(self.in_channels, self.kernel_size[0] * self.kernel_size[1] * self.out_channels)
        dLdX = (corrected_kernel @ unfolded).view(self.input.shape)

        return dLdX

    def params(self):
        if self.use_bias:
            return [self.weight, self.bias]
        else:
            return [self.weight]

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
        self.nearest_upsampling = NearestUpsampling(self.scale_factor)
        
        self.out_upsample = self.nearest_upsampling.forward(input_)
        self.out_conv = self.conv.forward(self.out_upsample)

        return self.out_conv
    
    def backward(self, grad):
        # Output: Z = Upsampling(Conv(X)) = Upsampling(Y), Y = Conv(X)
        # So, dLdX = dLdZ.dZdX = dLdZ.dZdY.dYdX
        
        grad_1 = self.conv.backward(grad)
        grad_2 = self.nearest_upsampling.backward(grad_1)
        
        return grad_2
        
    def params(self):
        return self.conv.params()


class NearestUpsampling(Upsampling):
    def __init__(self, scale_factor):
        """
        Store the attributes
        """
        super().__init__(scale_factor)

    def params(self):
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
        self.filter_ones = torch.empty(self.scale_factor**2, dtype = torch.float).zero_() + 1
        self.unfolded = unfold(grad, kernel_size = self.scale_factor, stride=self.scale_factor).to(torch.float).view(grad.shape[0],
                                                                                                                grad.shape[1], 
                                                                                                                self.scale_factor*self.scale_factor,
                                                                                                                grad.shape[2]//self.scale_factor*grad.shape[3]//self.scale_factor)
        wxb = self.filter_ones@self.unfolded
        actual = wxb.view(grad.shape[0], grad.shape[1], grad.shape[2]//self.scale_factor,grad.shape[3]//self.scale_factor)
        return actual

# Working space below

import torch

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
        train_input = train_input.to(torch.float)
        train_target = train_target.to(torch.float)
        train_input = train_input.to(self.device)
        train_target = train_target.to(self.device)
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

if __name__ == '__main__':
    x = torch.arange(150).view(2,3,5,5).to(torch.float)
    y = (torch.arange(24).view(2,3,2,2) + torch.normal(0, 1, (2,3,2,2))).to(torch.float)
    
    in_channels = 3
    out_channels = 3
    kernel_size = 3
    
    conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride = 2)
    conv_ours = Conv2d(in_channels, out_channels, kernel_size, stride = 2)

    conv.weight = torch.nn.Parameter(torch.ones_like(conv.weight))
    conv.bias = torch.nn.Parameter(torch.zeros_like(conv.bias))
    
    conv_ours.weight = torch.ones_like(conv_ours.weight)
    conv_ours.weight.grad = torch.zeros_like(conv_ours.weight)
    conv_ours.bias = torch.zeros_like(conv_ours.bias)
    conv_ours.bias.grad = torch.zeros_like(conv_ours.bias)
    
    
    if torch.allclose(conv.forward(x), conv_ours.forward(x)):
        print('CONGRATS! The output is the same.')
    else:
        print('OOPS, something is still not quite right. Look at both outputs below.')
        print(conv.forward(x))
        print(conv_ours.forward(x))
        
    loss = torch.nn.MSELoss()
    loss_ours = MSE()
    
    sgd = torch.optim.SGD([conv.weight, conv.bias], lr = 0.001)
    sgd_ours = SGD(conv_ours.params(), lr = 0.001)
    
    output = conv.forward(x)
    output_ours = conv_ours.forward(x)
        
    loss_val = loss.forward(output,y).requires_grad_()
    loss_val_ours = loss_ours.forward(output_ours,y)
    loss_val.backward()
    grad = loss_ours.backward()
    conv_ours.backward(grad)
    sgd.step()
    sgd_ours.step()
        
    if torch.allclose(conv.bias, conv_ours.bias):
        print('CONGRATS! The updated weights are identical.')
    else:
        print(f'OOPS, the weight didn\'t match after {iters} iterations. Their difference was {torch.norm(conv.weight - conv_ours.weight):.2f}')
        print(conv.weight)
        print(conv_ours.weight)    

    if torch.allclose(conv.bias, conv_ours.bias):
        print('CONGRATS! The updated biases are identical.')
    else:
        print('OOPS, something is still not quite right. Look at both biases below.')
        print(conv.bias)
        print(conv_ours.bias)