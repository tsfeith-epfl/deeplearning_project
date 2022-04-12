import torch
from torch import nn
from torchvision import transforms
import random
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
        # epochs and batch size are placeholders, might need more epochs and we may need to reduce batch size
        self.nb_epochs = 250
        self.mini_batch_size = 100 
        
        # All convolutions use padding mode “same”
        # CHECK ALL INPUT SIZES (OUTPUTS SHOULD BE CORRECT) - LOOK AT U-NET PAPER/U-NET IMPLEMENTATIONS
        # STARTED IT, IT'S NOT FINISHED, THERE ARE STILL SEVERAL LAYERS MISSING
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
        self.upsample5 = nn.Upsample(scale_factor=2)
        # after this we concatenate with result from pool4
        self.dec_conv5a = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.dec_conv5b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        

    def forward(self, x):
        
        # except for the last layer all convolutions are followed by leaky ReLU activation
        # function with alpha = 0.1. Other layers have linear activation. Upsampling is nearest-neighbor.
        
        # WRITE FORWARD PASS FOR A U-NET
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
                output = self.forward(train_input.narrow(0, b, self.mini_batch_size))
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
    
def data_augmentation(imgs_1, 
                      imgs_2,
                      hflip_prob = 0.5,
                      vflip_prob = 0.5,
                      gauss_kernel_size = 3,
                      sol_thresh = 150,
                      local_blur = True):
    H, W = imgs_1.shape[-2], imgs_1.shape[-1]
    
    global_augs = transforms.Compose([transforms.RandomHorizontalFlip(p=hflip_prob),
                                      transforms.RandomResizedCrop((H,W), scale = (0.5, 1.0)),
                                      transforms.RandomVerticalFlip(p=vflip_prob),
                                      transforms.GaussianBlur(kernel_size = gauss_kernel_size),
                                      transforms.RandomSolarize(threshold = sol_thresh)
                                     ])
    if local_blur:
        local_augs = transforms.Compose([transforms.RandomResizedCrop((H,W), scale = (0.05, 0.5)),
                                         transforms.GaussianBlur(kernel_size = 3),
                                        ])
    else:
        local_augs = transforms.Copse([transforms.RandomResizedCrop((H,W), scale = (0.05, 0.5))])
    
    # taking inspiration from DINO, we use several local crops and 1 global crop per image
    # Hopefully the global views will help the model to learn the general image modifications
    # while the local views will help it to learn small-scale modifications
    # DINO repo: https://github.com/facebookresearch/dino/blob/main/main_dino.py
    seed = random.randint(0,100)
    torch.manual_seed(seed)
    augs_1 = global_augs(imgs_1)
    for _ in range(4):
        augs_1 = torch.cat((augs_1, local_augs(imgs_1)))
    torch.manual_seed(seed)
    augs_2 = global_augs(imgs_2)
    for _ in range(4):
        augs_2 = torch.cat((augs_2, local_augs(imgs_2)))
    
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

# ----------- AUGMENTATIONS TESTS: EVERYTHING SEEMS TO BE FINE ----------------------
# ----------------------- DELETE AFTERWARDS -----------------------------------------

noisy_imgs_1, noisy_imgs_2 = torch.load('../data/train_data.pkl ')
noisy_imgs, clean_img = torch.load('../data/val_data.pkl ')

new_imgs_1, new_imgs_2 = data_augmentation(noisy_imgs_1[:1,:,:,:], noisy_imgs_2[:1,:,:,:])

print(new_imgs_1.shape)

for i in range(len(new_imgs_1)):
    cv2.imwrite(f'new_1_{i}.png', new_imgs_1[i].permute(1,2,0).cpu().numpy())
    cv2.imwrite(f'new_2_{i}.png', new_imgs_2[i].permute(1,2,0).cpu().numpy())
cv2.imwrite(f'noisy_1.png', noisy_imgs_1[0].permute(1,2,0).cpu().numpy())
cv2.imwrite(f'noisy_2.png', noisy_imgs_2[0].permute(1,2,0).cpu().numpy())

# ----------------------------------------------------------------------------------