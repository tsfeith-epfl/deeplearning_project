import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as F1
from torch import optim
from torchvision import transforms
import random

from others.gauss_noise import AddGaussianNoise

class Model(nn.Module):
    def __init__(self):
        """
        Instantiate model, optimizer, loss function, any other stuff needed.

        Returns
        -------
        None
        """
        super(Model, self).__init__()

        # Use a standard U-Net with Transposed Convolutions on the decoder branch
        self.enc_conv0 = nn.Conv2d(3, 48, kernel_size=3, padding='same')
        self.enc_conv1 = nn.Conv2d(48, 48, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv2 = nn.Conv2d(48, 48, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv3 = nn.Conv2d(48, 48, kernel_size=3, padding='same')
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv4 = nn.Conv2d(48, 48, kernel_size=3, padding='same')
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv5 = nn.Conv2d(48, 48, kernel_size=3, padding='same')
        self.pool5 = nn.MaxPool2d(kernel_size=2)
        self.enc_conv6 = nn.Conv2d(48, 48, kernel_size=3, padding='same')
        self.upsample5 = nn.ConvTranspose2d(48, 48, kernel_size=2, stride=2, padding=0)
        self.dec_conv5a = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.dec_conv5b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample4 = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2, padding=0)
        self.dec_conv4a = nn.Conv2d(144, 96, kernel_size=3, padding='same')
        self.dec_conv4b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample3 = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2, padding=0)
        self.dec_conv3a = nn.Conv2d(144, 96, kernel_size=3, padding='same')
        self.dec_conv3b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample2 = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2, padding=0)
        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample1 = nn.ConvTranspose2d(96, 96, kernel_size=2, stride=2, padding=0)
        self.dec_conv1a = nn.Conv2d(99, 64, kernel_size=3, padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.dec_conv1 = nn.Conv2d(32, 3, kernel_size=3, padding='same')
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr = 1e-3)

        # Mini-Batch found to be empirically the best
        self.mini_batch_size = 625
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)
        
        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
                module.bias.data.zero_()
                    
    def forward(self, x, sharpen = True, sharp_factor = 1.):
        
        # add skip connections
        skip_connects = [x]
        
        # negative slope for leaky ReLU following original paper
        x = F.leaky_relu(self.enc_conv0(x), negative_slope = 0.1)
        x = F.leaky_relu(self.enc_conv1(x), negative_slope = 0.1)
        x = self.pool1(x)
        skip_connects.append(x)
        
        x = F.leaky_relu(self.enc_conv2(x), negative_slope = 0.1)
        x = self.pool2(x)
        skip_connects.append(x)
        
        x = F.leaky_relu(self.enc_conv3(x), negative_slope = 0.1)
        x = self.pool3(x)
        skip_connects.append(x)
        
        x = F.leaky_relu(self.enc_conv4(x), negative_slope = 0.1)
        x = self.pool4(x)
        skip_connects.append(x)
        
        x = F.leaky_relu(self.enc_conv5(x), negative_slope = 0.1)
        x = self.pool5(x)
        x = F.leaky_relu(self.enc_conv6(x), negative_slope = 0.1)
        
        # ---------------------------------------------------
        x = self.upsample5(x)
        x = torch.cat((x, skip_connects.pop()), dim=1)
        x = F.leaky_relu(self.dec_conv5a(x), negative_slope = 0.1)
        x = F.leaky_relu(self.dec_conv5b(x), negative_slope = 0.1) 

        x = self.upsample4(x)
        x = torch.cat((x, skip_connects.pop()), dim=1)
        x = F.leaky_relu(self.dec_conv4a(x), negative_slope = 0.1)
        x = F.leaky_relu(self.dec_conv4b(x), negative_slope = 0.1)
        
        x = self.upsample3(x)
        x = torch.cat((x, skip_connects.pop()), dim=1)
        x = F.leaky_relu(self.dec_conv3a(x), negative_slope = 0.1)
        x = F.leaky_relu(self.dec_conv3b(x), negative_slope = 0.1)
        
        x = self.upsample2(x)
        x = torch.cat((x, skip_connects.pop()), dim=1)
        x = F.leaky_relu(self.dec_conv2a(x), negative_slope = 0.1)
        x = F.leaky_relu(self.dec_conv2b(x), negative_slope = 0.1)
        
        x = self.upsample1(x)
        x = torch.cat((x, skip_connects.pop()), dim=1)
        x = F.leaky_relu(self.dec_conv1a(x), negative_slope = 0.1)
        x = F.leaky_relu(self.dec_conv1b(x), negative_slope = 0.1)

        x = F.relu(self.dec_conv1(x))
        
        # application of explicit sharpening operator
        if sharpen:
            x = F.relu(x + sharp_factor*(x - F1.gaussian_blur(x, 3)))
        
        x = x - F.relu(x - 255)
        
        return x

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
        self.load_state_dict(model)
        self.to(self.device)

    def train(self,
              train_input,
              train_target,
              num_epochs,
              use_crops = False,
              n_local_crops = 2,
              sharpen = True,
              sharpen_factor = 1.):
        """
        Train the model.

        Parameters
        ----------
        train_input : torch.Tensor
            Tensor of size (N, C, H, W) containing a noisy version of the images
        train_target : torch.Tensor
            Tensor of size (N, C, H, W) containing another noisy version of the
            same images
        num_epochs : int
            Number of epochs to use for training
        use_crops : bool
            Whether or not to use (local and global) crops as part of the data
            augmentation strategy.
        n_local_crops : int
            Number of local crops to use. Only relevant if use_crops = True
        sharpen : bool
            Whether or not to apply the explicit sharpening operator
        sharpen_factor : float
            Sharpening factor to apply as the last step of the networ. Only 
            relevant if sharpen = True

        Returns
        -------
        None
        """
        
        train_input = train_input.to(torch.float)
        train_target = train_target.to(torch.float)
        train_input = train_input.to(self.device)
        train_target = train_target.to(self.device)
        train_input.requires_grad_()
        train_target.requires_grad_()
        
        
        n_samples = len(train_input)*(1 + n_local_crops) if use_crops else len(train_input) 
        print('\nTRAINING STARTING...')
        for e in range(num_epochs):
            epoch_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                self.optimizer.zero_grad()
                train = train_input.narrow(0, b, self.mini_batch_size)
                target = train_target.narrow(0, b, self.mini_batch_size)
                train, target = data_augmentations(train, target, use_crops = use_crops, n_local_crops = n_local_crops)
                output = self.forward(train, sharpen, sharpen_factor)
                loss = self.criterion(output, target).requires_grad_()
                epoch_loss += loss.item()/n_samples
                loss.backward()
                self.optimizer.step()
            print(f"Epoch {e+1}: Loss = {epoch_loss};")
        
    def predict(self, test_input, sharpen = True, sharpen_factor = 1.):
        """
        Perform inference.

        Parameters
        ----------
        test_input : torch.Tensor
            Tensor of size (N1, C, H, W) that has to be denoised by the trained
            or loaded network.
        sharpen : bool
            Whether or not to apply the explicit sharpening operator
        sharpen_factor : float
            Sharpening factor to apply as the last step of the networ. Only 
            relevant if sharpen = True

        Returns
        -------
        denoised_signal: torch.Tensor
            Tensor of size (N1, C, H, W) containing the denoised signal.
        """
        test_input = test_input.to(torch.float)
        test_input = test_input.to(self.device)
        return self.forward(test_input, sharpen, sharpen_factor)
        
    
def data_augmentations(imgs_1, 
                       imgs_2,
                       hflip_prob = 0.5,
                       vflip_prob = 0.5,
                       use_crops = False,
                       n_local_crops = 2):
    H, W = imgs_1.shape[-2], imgs_1.shape[-1]
    
    # By default flips and gaussian noise are always applied
    # Crops may or may not be used
    if use_crops:
        global_augs = transforms.Compose([transforms.RandomHorizontalFlip(p=hflip_prob),
                                          transforms.RandomVerticalFlip(p=vflip_prob),
                                          transforms.RandomResizedCrop((H,W), scale = (0.7, 1.0)),
                                          AddGaussianNoise(0., 5.)
                                          ])

        local_augs = transforms.Compose([transforms.RandomHorizontalFlip(p=hflip_prob),
                                         transforms.RandomVerticalFlip(p=vflip_prob),
                                         transforms.RandomResizedCrop((H,W), scale = (0.05, 0.4)),
                                         AddGaussianNoise(0., 5.)
                                         ])
    else:
        global_augs = transforms.Compose([transforms.RandomHorizontalFlip(p=hflip_prob),
                                          transforms.RandomVerticalFlip(p=vflip_prob),
                                          AddGaussianNoise(0., 5.)
                                          ])
        n_local_crops = 0
        
    # We use several local crops and 1 global crop per image
    # Hopefully the global views will help the model to learn the general image modifications
    # while the local views will help it to learn small-scale modifications
    seed = random.randint(0,100)
    
    torch.manual_seed(seed)
    augs_1 = global_augs(imgs_1)
    for _ in range(n_local_crops):
        augs_1 = torch.cat((augs_1, local_augs(imgs_1)))
    
    torch.manual_seed(seed)
    augs_2 = global_augs(imgs_2)
    for _ in range(n_local_crops):
        augs_2 = torch.cat((augs_2, local_augs(imgs_2)))
    
    return augs_1, augs_2
    
def psnr(denoised, ground_truth):
    """
    Compute the peak signal to noise ratio.

    Parameters
    ----------
    denoised : torch.Tensor
        Tensor of size (N1, C, H, W) containing the denoised signal resulting
        from the model.

    ground_truth : torch.Tensor
        Tensor of size (N1, C, H, W) containing the true denoised signal.

    Returns
    -------
    psnr : float
        Value of PSNR.

    """ 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ground_truth = ground_truth.to(device)
    denoise = denoised.to(device)
    mse = torch.mean((denoised-ground_truth)**2)
    psnr = 20*torch.log10(torch.max(denoised)) - 10*torch.log10(mse+10**-8)
    return psnr

if __name__ == '__main__':

    from pathlib import Path
    data_path = Path(__file__).parent
    
    noisy_imgs_1, noisy_imgs_2 = torch.load(data_path / 'train_data.pkl')
    noisy_imgs, clean_imgs = torch.load(data_path / 'val_data.pkl')
    print('DATA IMPORTED')

    model = Model()
    model.to(model.device)
    model.load_pretrained_model()
    # model.train(noisy_imgs_1, noisy_imgs_2, 60)

    output = model.predict(noisy_imgs)
    print(f'PSNR: {psnr(output/255, clean_imgs/255)} dB')