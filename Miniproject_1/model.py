import torch
from torch import nn

class Model ():
    def __init__(self):
        """
        Instantiate model, optimizer, loss function, any other stuff needed.

        Returns
        -------
        None
        """
        super().__init__()
        
        # try different criterion (like MSELoss or NLLLoss or L1Loss), optimizer (like Adam or ASGD)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(model.parameters(), lr = 1e-1)
        # epochs and batch size and placeholders, might need more epochs and we may need to reduce batch size
        self.nb_epochs = 250
        self.mini_batch_size = 100 
        
        # don't use this! Instead mimic what they used in the paper, then if we have time we can work from there
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=2)
        self.fc1 = nn.Linear(9 * 64, nb_hidden)
        self.fc2 = nn.Linear(nb_hidden, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc1(x.view(-1, 9 * 64)))
        x = self.fc2(x)
        return x

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

        for e in range(self.nb_epochs):
            for b in range(0, train_input.size(0), self.mini_batch_size):
                output = model(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, mini_batch_size))
                model.zero_grad()
                loss.backward()
                self.optimizer.step()
        
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

def psnr (denoised, ground_truth):
    """
    Compute the peak signal to noise ratio.

    Parameters
    ----------
    denoised : torch.Tensor
        Tensor of size (N1, C, H, W) containing the denoised signal resulting
        from the model. Must have range [0, 1].

    ground_truth : torch.Tensor
        Tensor of size (N1, C, H, W) containing the true denoised signal. Must
        have range [0, 1].

    Returns
    -------
    psnr : float
        Value of peak signal to noise ratio.

    """

    mse = torch.mean((denoised-ground_truth)**2)
    psnr = -10*torch.log10(mse+10**-8)
    return psnr

def transform_imgs(imgs_1, imgs_2):
    # for these transformations doesn't make much sense to add noise
    # we can try croppings, gaussian, solarization, flips, rotations,....
    # I still need to figure out how to apply exactly the same transformations to the input and the target
    pass

noisy_imgs_1, noisy_imgs_2 = torch.load ('train_data.pkl ')
noisy_imgs, clean_img = torch.load ('val_data.pkl ')
