import torch

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()).to(self.device) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)