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
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr
    
    def step(self):
        for p in self.params: p -= lr*grad?
        


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


class MSE(object):
    def forward(self, *args):
        raise NotImplementedError
    def backward(self, *gradwrtoutput):
        raise NotImplementedError
    def param(self):
        return []


    
    def mse(self, prediction, target):
        """
        Perform mean squared error given targets and predictions.
        """
        return torch.sum((prediction - target)**2)/prediction 
    
    def relu(self, activation):
        return max(0, activation)
    
    def sigmoid(self, activation):
        return 1/(1+exp(-activation))
