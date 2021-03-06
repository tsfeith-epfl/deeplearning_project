from torch.nn.functional import fold, unfold
import math
import pickle
import torch
torch.set_grad_enabled(False)


class ReLU():

    def __init__(self):
        """
        Set the name and the device for the module
        """
        self.name = "RelU"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_):
        """
        ReLU(x) = max(0, x): returns the max between 0 and the input
        """
        input_ = input_.to(self.device)
        input_[input_<torch.empty(input_.shape).zero_().to(self.device)] = 0
        return input_.to('cpu')
    
    def backward(self, grad):
        """
        Derivative of ReLU: 1 if input > 0, 0 elsewhere
        """
        grad = grad.to(self.device)
        zeros = torch.empty(grad.shape).zero_().to(self.device)
        zeros[grad > zeros] = 1
        return zeros.to('cpu')

    def params(self):
        """
        In this case there are no parameters
        """
        return []
    
    def to(self, device):
        pass

class Sigmoid():

    def __init__(self):
        """
        Set the name and device for the module
        """
        self.name = "Sigmoid"
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, input_):
        """
        Sigmoid(x) = 1/(1+e^(-x))
        """
        input_ = input_.to(self.device)
        return (1 / (1 + (-input_).exp())).to('cpu')
    
    def backward(self, grad):
        """
        Derivative of sigmoid: dsig(x)/dx = sig(x)(1-sig(x))
        """
        grad = grad.to(self.device)
        return (1 / (1 + (-grad).exp()) * (1 - 1 / (1 + (-grad).exp()))).to('cpu')
    
    def params(self):
        """
        In this case there are no parameters
        """
        return []
       
    def to(self, device):
        pass

class MSE():
    """
    Mean Squared Error
    """
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    def forward(self, predictions, targets):
        """
        Mean Squared Error: MSE(x) = 1/N * sum((y - f(x))^2)
        """
        self.predictions, self.targets = predictions.to(self.device), targets.to(self.device)
        return ((self.predictions - self.targets)**2).mean().to('cpu')
        
    def backward(self):
        """
        Derivative of MSE = 2/N * (y - f(x))
        """
        return (2/self.predictions.shape[0] * (self.predictions - self.targets)).to('cpu')

class SGD():
    """
    Stochastic Gradient Descent optimizer
    """
    def __init__(self, params, lr):
        """
        Initialize: model's parameters and learning rate
        """
        self.params = params
        self.lr = lr
    
    def step(self):
        """
        Perform one step of Stochastig Gradient Descent
        """
        for param in self.params:
            param -= self.lr*param.grad

    def zero_grad(self):
        """
        Zero all the gradients
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for param in self.params: param.grad = torch.empty(param.shape).zero_().to(device)


class Sequential():
    """
    Container class for the different modules
    """
    def __init__(self, *args):
        """
        Initialize an empty list in which we're going to append the modules
        """
        self.layers = args
        self.model = []
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for module in args:
            self.model.append(module)
            
    def forward(self, input_):
        """
        Do the forward pass of each module and keep track of the output
        """
        self.input = input_.to(self.device)
        for module in self.model:
            self.input = module.forward(self.input)    
        
        return self.input.to('cpu')
    
    def backward(self, grad):
        """
        Do the backward pass of each module and keep track of the gradient
        """
        grad = grad.to(self.device)
        for module in self.model[::-1]:
            grad = module.backward(grad)

        return grad.to('cpu')

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
            # OUT = [(IN???D(K-1)+2P-1)/S]+1
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
        
        # Initialize the weigths according to a uniform distribution
        self.weight = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).uniform_(-k, k).to(self.device)
        self.weight.grad = torch.empty(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1]).zero_().to(self.device)
        
        if self.use_bias:
            # Initialize the biases according to a uniform distribution
            self.bias = torch.empty(self.out_channels).uniform_(-k, k).to(self.device)
            self.bias.grad = torch.empty(self.out_channels).zero_().to(self.device)
            
    def forward(self, input_):
        """
        Perform convolution as a linear transformation
        """
        self.input = input_.to(self.device)
        self.output_shape = (math.floor((self.input.shape[2] + 2*self.padding[0] - self.dilation[0]*(self.kernel_size[0] - 1) - 1)/self.stride[0] + 1),
                             math.floor((self.input.shape[3] + 2*self.padding[1] - self.dilation[1]*(self.kernel_size[1] - 1) - 1)/self.stride[1]  + 1))
        
        unfolded = unfold(self.input, kernel_size = self.kernel_size,  dilation=self.dilation, padding=self.padding, stride=self.stride).to(self.device)
        
        # save unfolded to use it in the backward pass
        self.unfolded = unfolded

        if self.use_bias:
            wxb = self.weight.view(self.out_channels, -1) @ unfolded + self.bias.view(1, -1, 1)
        else:
            wxb = self.weight.view(self.out_channels, -1) @ unfolded
            
        wxb = wxb.to(self.device)
        actual = wxb.view(input_.shape[0], self.out_channels, self.output_shape[0], self.output_shape[1]).to(self.device)
        
        return actual.to('cpu')
        
    def backward(self, grad):
        """
        Compute gradients wrt parameters(w, b) and input(x)
        """
        
        # compute dLdW
        self.grad = grad.to(self.device)
        
        actual = self.grad.view(self.input.shape[0], self.out_channels, -1).bmm(self.unfolded.transpose(1,2)).sum(dim = 0).view(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])

        self.weight.grad.add_(actual)

        # compute the gradient dLdb
        self.bias.grad += self.grad.sum((0,2,3))

        # compute the gradient dLdX
        input_ =  self.weight.view(-1, self.out_channels) @ self.grad.view(self.input.shape[0],self.out_channels,-1)
        dLdX = fold(input_, kernel_size = self.kernel_size, output_size = (self.input.shape[2],self.input.shape[3]),stride = self.stride,padding = self.padding)

        return dLdX.to('cpu')

    def params(self):
        """
        Store the parameters
        """
        return [self.weight, self.bias]

    def to(self, device):
        for param in self.params():
            param.to(device)

class Upsampling():
    def __init__(self, scale_factor, in_channels, out_channels, kernel_size = 3, padding=1):
        """
        Store the attributes
        """
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.kernel_size = kernel_size 
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.conv = Conv2d(self.in_channels, self.out_channels, kernel_size=self.kernel_size, stride = 1, padding = self.padding)
        self.conv.to(self.device)
        self.weight, self.bias = self.conv.params()
        self.nearest_upsampling = NearestUpsampling(self.scale_factor)
        self.name = "Upsampling"
        
    def forward(self, input_):
        """
        perform upsampling using nearest neighbor rule and then convolution, to have a transposed convolution
        """
        self.input = input_.to(self.device)
        kernel_size = (self.scale_factor, self.scale_factor)
        
        self.out_upsample = self.nearest_upsampling.forward(input_)
        self.out_conv = self.conv.forward(self.out_upsample)

        return self.out_conv.to('cpu')
    
    def backward(self, grad):
        """
        Combine the backward passes of Conv2d and NearestUpsampling
        """
        grad = grad.to(self.device)
        grad_1 = self.conv.backward(grad)
        grad_2 = self.nearest_upsampling.backward(grad_1)
        
        return grad_2.to('cpu')
        
    def params(self):
        """
        Store the parameters
        """
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
        
    def forward(self, input_):
        """
        Perform upsampling using nearest neighbor rule
        """
        input_ = input_.to(self.device)
        return input_.repeat_interleave(self.scale_factor,3).repeat_interleave(self.scale_factor,2).to('cpu')
    
    def backward(self, grad):
        """
        Convolve the gradient with a filter of ones to return the correct value
        """
        grad = grad.to(self.device)
        self.filter_ones = (torch.empty(self.scale_factor**2, dtype = torch.float).zero_() + 1).to(self.device)
        unfolded = unfold(grad, kernel_size = self.scale_factor, stride=self.scale_factor).to(torch.float).view(grad.shape[0],
                                                                                                                grad.shape[1], 
                                                                                                                self.scale_factor*self.scale_factor,
                                                                                                                grad.shape[2]//self.scale_factor*grad.shape[3]//self.scale_factor).to(self.device)
        wxb = (self.filter_ones@unfolded).to(self.device)
        actual = wxb.view(grad.shape[0], grad.shape[1], grad.shape[2]//self.scale_factor,grad.shape[3]//self.scale_factor)
        return actual.to('cpu')
    
    def params(self):
        """
        Store the attributes
        """
        return []

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

    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel.pth into the model.

        Returns
        -------
        None
        """
        from pathlib import Path
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path,'rb') as infile:
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
              num_epochs):
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
        train_input = train_input / 255
        train_target = train_target / 255
        print('\nTRAINING STARTING...')
        n_samples = train_input.shape[0]
        for e in range(num_epochs):
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
        test_input = test_input.float()
        return (self.model.forward(test_input/255)*255).to('cpu')

if __name__ == '__main__':
    
    
    from pathlib import Path
    # data_path = Path(__file__).parent
    
    noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl')
    noisy_imgs, clean_imgs = torch.load('val_data.pkl')
    print('DATA IMPORTED')
    
    noisy_imgs_1 = noisy_imgs_1
    noisy_imgs_2 = noisy_imgs_2

    model = Model()
#     model.load_pretrained_model()
    model.train(noisy_imgs_1, noisy_imgs_2, 1)
    model.save_model()

    # output = model.predict(noisy_imgs)
    # print(f'PSNR: {psnr(output/255, clean_imgs/255)} dB')

