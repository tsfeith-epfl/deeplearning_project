import torch
import math
import pickle
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

    def __init__(self):
        """
        Set the name of the module
        """
        self.name = "RelU"

    def forward(self, input_):
        """
        ReLU(x) = max(0, x): returns the max between 0 and the input
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        input_[input_<torch.empty(input_.shape).zero_().to(device)]= 0.001 * input_[input_<torch.empty(input_.shape).zero_().to(device)]
        return input_
    
    def backward(self, gradwrtoutput):
        """
        Derivative of ReLU: 1 if input > 0, 0 elsewhere
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        zeros = torch.empty(gradwrtoutput.shape).zero_().to(device)
        mask = gradwrtoutput > zeros
        zeros[mask] = 1
        zeros[~mask] = 0.001
        return zeros

    def params(self):
        return []
    
    def to(self, device):
        pass

class Sigmoid():

    def __init__(self):
        """
        Set the name of the module
        """
        self.name = "Sigmoid"

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
       
    def to(self, device):
        pass

# # LOSS FUNCTIONS

class MSE():
    def __init__(self):
        pass
    def forward(self, predictions, targets):
        """
        Mean Squared Error: MSE(x) = 1/N * (y - f(x))^2
        """
        self.predictions, self.targets = predictions, targets
        return ((self.predictions - self.targets)**2).sum()
        
    def backward(self):
        """
        Derivative of MSE = 2/N * (y - f(x))
        """
        return 2/self.predictions.numel() * (self.predictions - self.targets)

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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for param in self.params: param.grad = torch.empty(param.shape).zero_().to(device)


class Sequential(): #I may need also functions 
    def __init__(self, *args):
        """
        Initialize an empty list in which we're going to append the modules
        """
        self.layers = args
        self.model = []
        for module in args:
            self.model.append(module)
            
    def forward(self, input_):
        """
        Do the forward pass of each module and keep track of the output
        """
        self.input = input_
        for module in self.model:
            self.input = module.forward(self.input)    
        
        return self.input
    
    def backward(self, gradwrtoutput):
        """
        Do the backward pass of each module and keep track of the gradient
        """
        grad = gradwrtoutput
        for module in self.model[::-1]:
            grad = module.backward(grad)
            
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
    
    def to(self, device):
        if device == 'cuda' or device == 'cpu':
            try:
                self.input.to(device)
            except:
                pass
            for module in self.model:
                module.to(device)


# +
from torch.nn.functional import fold, unfold
##ADD weights initialization

from torch.nn.functional import fold, unfold
##ADD weights initialization

class Conv2d():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias = True):
        """
        Store the attributes and initialize the parameters and gradient tensors
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.name = "Conv2d"

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
            pad0 = (self.stride[0] * (self.in_channels - 1) - self.in_channels + self.dilation[0] * (self.kernel_size[0] - 1))//2
            pad1 = (self.stride[1] * (self.in_channels - 1) - self.in_channels + self.dilation[1] * (self.kernel_size[1] - 1))//2
            self.padding = (pad0, pad1)
        elif padding == 'valid':
            self.padding = (0, 0)
        else:
            raise Exception("Please enter padding parameters as tuple or int, or a string in {\"same\", \"valid\"}")

        k = math.sqrt(1/(self.in_channels*self.kernel_size[0]*self.kernel_size[1]))

        self.use_bias = bias
        self.weight = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k, k).to(self.device)
        self.weight.grad = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).zero_().to(self.device)
        
        if self.use_bias:
            self.bias = torch.empty(self.out_channels).uniform_(-k, k).to(self.device)
            self.bias.grad = torch.empty(self.out_channels).zero_().to(self.device)
            
    def forward(self, input_):
        """
        Perform convolution as a linear transformation
        """
        self.input = input_
        self.output_shape = (math.floor((self.input.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                             math.floor((self.input.shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
        output = torch.empty(self.input.shape).to(self.device)
        unfolded = unfold(input_, kernel_size = self.kernel_size,  dilation=self.dilation, padding=0, stride=self.stride).to(self.device)
        self.unfolded = unfolded
        
        if self.use_bias:
            wxb = self.weight.view(self.out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)
        else:
            wxb = self.weight.view(self.out_channels, -1) @ unfolded
        wxb = wxb.to(self.device)
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
        self.grad = grad

        actual = grad.view(self.input.shape[0], self.out_channels, -1).bmm(self.unfolded.transpose(1,2)).sum(dim = 0).view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        self.weight.grad.add_(actual)#with b=1 .mean(0) makes a mess

        # compute the gradient dLdb
        self.bias.grad += self.grad.sum((0,2,3))

        # compute the gradient dLdX
        kernel_mirrored = self.weight.flip([2,3]).to(self.device)

        expanded_grad = torch.empty(self.input.shape[0], self.out_channels, (grad.shape[2]-1) * (self.stride[0] - 1) + grad.shape[2], (grad.shape[3]-1) * (self.stride[1] - 1) + grad.shape[3]).zero_().to(self.device)
        expanded_grad[:, :, ::self.stride[0], ::self.stride[1]] = self.grad

        unfolded = unfold(expanded_grad, kernel_size = self.kernel_size, padding = (self.kernel_size[0] - 1, self.kernel_size[1] - 1)).to(self.device)

        corrected_kernel = kernel_mirrored.view(self.in_channels, self.kernel_size[0] * self.kernel_size[1] * self.out_channels).to(self.device)
        dLdX = (corrected_kernel @ unfolded).view(self.input.shape).to(self.device)

        return dLdX

    def params(self):
        return [self.weight, self.bias]

    def to(self, device):
        for param in self.params():
            param.to(device)

class Upsampling():
    def __init__(self, scale_factor, in_channels, out_channels):
        """
        Store the attributes
        """
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride = 1, padding = 0)
        self.weight, self.bias = self.conv.params()
        self.nearest_upsampling = NearestUpsampling(self.scale_factor)
        self.name = "Upsampling"
        
    def forward(self, input_):
        """
        perform upsampling using nearest neighbor rule and then convolution, to have a transposed convolution
        """
        self.input = input_ 
        kernel_size = (self.scale_factor, self.scale_factor)
        
        self.out_upsample = self.nearest_upsampling.forward(input_)
        self.out_conv = self.conv.forward(self.out_upsample)

        return self.out_conv
    
    def backward(self, grad):
        grad_1 = self.conv.backward(grad)
        grad_2 = self.nearest_upsampling.backward(grad_1)
        
        return grad_2
        
    def params(self):
        return [self.weight, self.bias]

    def to(self, device):
        for param in self.conv.params():
            param.to(device)
    

class NearestUpsampling():
    def __init__(self, scale_factor):
        """
        Store the attributes
        """
        self.scale_factor = scale_factor
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        self.filter_ones = (torch.empty(self.scale_factor**2, dtype = torch.float).zero_() + 1).to(self.device)
        unfolded = unfold(grad, kernel_size = self.scale_factor, stride=self.scale_factor).to(torch.float).view(grad.shape[0],
                                                                                                                grad.shape[1], 
                                                                                                                self.scale_factor*self.scale_factor,
                                                                                                                grad.shape[2]//self.scale_factor*grad.shape[3]//self.scale_factor).to(self.device)
        wxb = (self.filter_ones@unfolded).to(self.device)
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
        self.model = Sequential(Conv2d(in_channels=3, out_channels=48, kernel_size=2, stride=2),
                                ReLU(),
                                Conv2d(in_channels=48, out_channels=48, kernel_size=2, stride=2),
                                ReLU(),
                                Upsampling(scale_factor=2, in_channels=48, out_channels=24),
                                ReLU(),
                                Upsampling(scale_factor=2, in_channels=24, out_channels=3),
                                Sigmoid())
        self.optimizer = SGD(self.model.params(), 1e-6)
        self.criterion = MSE()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

        self.mini_batch_size = 100

    def save_model(self):
      """
      This saves the parameters in bestmodel.pth.

      Returns
      -------
      None
      """
      state_dict = {}  
      for i,layer in enumerate(self.model.layers):
        if len(layer.params())==2:
          state_dict[str(i)+"."+layer.name] = [layer.params()[0], layer.params()[1]]
        elif len(layer.params())==1:
          state_dict[str(i)+"."+layer.name] = layer.params()[0]
        else:
          state_dict[str(i)+"."+layer.name] = []

      outfile = open("bestmodel.pth",'wb')
      pickle.dump(state_dict,outfile)
      print(state_dict)
    # Working space below

    def load_pretrained_model(self):
      """
      This loads the parameters saved in bestmodel.pth into the model.

      Returns
      -------
      None
      """
      infile = open("bestmodel.pth",'rb')
      params = pickle.load(infile)
      for i,layer in enumerate(self.model.layers):
        layer_params = params[str(i)+"."+layer.name]
        if len(layer_params)==1:
          layer.weight.copy_(layer_params[0])
        elif len(layer_params)==2:
          layer.bias.copy_(layer_params[1])

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
        n_samples = train_input.numel()
        for e in range(epochs):
            epoch_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                self.optimizer.zero_grad()
                train = train_input.narrow(0, b, self.mini_batch_size)
                target = train_target.narrow(0, b, self.mini_batch_size)
                output = self.model.forward(train)
                loss = self.criterion.forward(output, target)
                epoch_loss += loss.item()/n_samples
                grad = self.criterion.backward()
                self.model.backward(grad)
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
        return self.model.forward(test_input)*255

if __name__ == '__main__':
    # Extensive series of tests
    """
    batch = 100
    in_channels = 3
    out_channels = 46
    kernel_size = 3
    in_size = 32
    out_size = int((in_size - kernel_size) / 2 + 1)
    
    x = torch.arange(batch*in_channels*in_size**2).view(batch,in_channels,in_size,in_size).to(torch.float)
    y = (torch.arange(batch*out_channels*out_size**2).view(batch,out_channels,out_size,out_size) + torch.normal(0, 1, (batch,out_channels,out_size,out_size))).to(torch.float)
    
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
        
    if torch.allclose(conv.weight, conv_ours.weight):
        print('CONGRATS! The updated weights are identical.')
    else:
        print('OOPS, something is still not quite right. Look at both weights below.')
        print(conv.weight)
        print(conv_ours.weight)    

    if torch.allclose(conv.bias, conv_ours.bias):
        print('CONGRATS! The updated biases are identical.')
    else:
        print('OOPS, something is still not quite right. Look at both biases below.')
        print(conv.bias)
        print(conv_ours.bias)
    
    """
    # model = Model()
    # out = model.predict(torch.rand(1, 3, 512, 512) * 255)
    # print(out.shape)
    
    from pathlib import Path
    # data_path = Path(__file__).parent
    
    noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl')
    noisy_imgs, clean_imgs = torch.load('val_data.pkl')
    print('DATA IMPORTED')
    
    noisy_imgs_1 = noisy_imgs_1 / 255
    noisy_imgs_2 = noisy_imgs_2 / 255

    model = Model()
#     model.load_pretrained_model()
    model.train(noisy_imgs_1, noisy_imgs_2, 1)
    model.save_model()

    # output = model.predict(noisy_imgs)
    # print(f'PSNR: {psnr(output/255, clean_imgs/255)} dB')
    
