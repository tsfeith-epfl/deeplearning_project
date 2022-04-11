import torch
from torch import nn
from torchvision import transforms
import random
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import cv2

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
    
def data_augmentation(imgs_1, imgs_2):
    # I think that by setting the seed we apply the same transformations to both image sets, still need to verify
    
    H, W = imgs_1.shape[2], imgs_1.shape[3]
    
    augs = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                               transforms.RandomResizedCrop((H,W), scale = (0.3, 1.0)),
                               #transforms.RandomRotation(90, fill = True),
                               transforms.RandomVerticalFlip(p=0.5),
                               transforms.GaussianBlur(kernel_size = 5),
                               transforms.RandomSolarize(threshold = 130)])
    
    seed = random.randint(0,100)
    torch.manual_seed(seed)
    augs_1 = augs(imgs_1)
    torch.manual_seed(seed)
    augs_2 = augs(imgs_2)
    
    return augs_1, augs_2
    
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

noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pkl ')
noisy_imgs, clean_img = torch.load('../data/val_data.pkl ')

new_imgs_1, new_imgs_2 = data_augmentation(noisy_imgs_1[:10,:,:,:], noisy_imgs_2[:10,:,:,:])

# plt.imview(noisy_imgs_1[0,:,:,:].permute(1,2,0))
print(noisy_imgs_1[0])
cv2.imwrite('noisy_1.png', noisy_imgs_1[0].permute(1,2,0).cpu().numpy())
cv2.imwrite('noisy_2.png', noisy_imgs_2[0].permute(1,2,0).cpu().numpy())
cv2.imwrite('new_1.png', new_imgs_1[0].permute(1,2,0).cpu().numpy())
cv2.imwrite('new_2.png', new_imgs_2[0].permute(1,2,0).cpu().numpy())

#save_image(noisy_imgs_1[0], 'noisy_1.png')
#save_image(noisy_imgs_2[0], 'noisy_2.png')
#save_image(new_imgs_1[0], 'new_1.png')
#save_image(new_imgs_2[0], 'new_2.png')