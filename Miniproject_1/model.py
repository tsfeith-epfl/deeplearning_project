import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
import random
import cv2

class Model(nn.Module):
    def __init__(self):
        """
        Instantiate model, optimizer, loss function, any other stuff needed.

        Returns
        -------
        None
        """
        super(Model, self).__init__()
        
        # All convolutions use padding mode “same”
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
        self.dec_conv5a = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.dec_conv5b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.dec_conv4a = nn.Conv2d(144, 96, kernel_size=3, padding='same')
        self.dec_conv4b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.dec_conv3a = nn.Conv2d(144, 96, kernel_size=3, padding='same')
        self.dec_conv3b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample2 = nn.Upsample(scale_factor=2)
        self.dec_conv2a = nn.Conv2d(144, 96, kernel_size=3, padding='same')
        self.dec_conv2b = nn.Conv2d(96, 96, kernel_size=3, padding='same')
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dec_conv1a = nn.Conv2d(99, 64, kernel_size=3, padding='same')
        self.dec_conv1b = nn.Conv2d(64, 32, kernel_size=3, padding='same')
        self.dec_conv1 = nn.Conv2d(32, 3, kernel_size=3, padding='same')
        
        
        # try different criterion (like MSELoss or L1Loss), optimizer (like Adam or ASGD)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr = 1e-1)
        # epochs and batch size are placeholders, might need more epochs and we may need to reduce batch size
        self.nb_epochs = 150
        self.mini_batch_size = 25 
        

    def forward(self, x):
        
        # except for the last layer all convolutions are followed by leaky ReLU activation
        # function with alpha = 0.1. Other layers have linear activation. Upsampling is nearest-neighbor.
        skip_connects = [x]
                
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

        x = self.dec_conv1(x)

        return x

    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel.pth into the model.

        Returns
        -------
        None
        """
        best_state_dict = torch.load('bestmodel.pth')
        self.load_state_dict(best_state_dict)

    def train(self, train_input, train_target, use_augs = True):
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
        # Not entirely sure if we can do this, but I don't think we can augment the input with operations that change the
        # pixel positions without augmenting the target. But still, we can train and see which one yields better.
        if use_augs:
            train_input, train_target = data_augmentations(train_input, train_target)
        for e in range(self.nb_epochs):
            if (e+1) % 10 == 0:
                print(f'EPOCH {e+1}')
            for b in range(0, train_input.size(0), self.mini_batch_size):
                self.optimizer.zero_grad()
                output = self.forward(train_input.narrow(0, b, self.mini_batch_size))
                loss = self.criterion(output, train_target.narrow(0, b, self.mini_batch_size))
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
        return self.forward(test_input)
        
    
def data_augmentations(imgs_1, 
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

    # it might be worthwhile to touch up this function a bit, improve its generability
    
    mse = torch.mean((denoised-ground_truth)**2)
    psnr = -10*torch.log10(mse+10**-8)
    return psnr

noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl ')
noisy_imgs, clean_imgs = torch.load('val_data.pkl ')
noisy_imgs_1 = noisy_imgs_1.to(torch.float)
noisy_imgs_2 = noisy_imgs_2.to(torch.float)
noisy_imgs = noisy_imgs.to(torch.float)
clean_imgs = clean_imgs.to(torch.float)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
noisy_imgs_1 = noisy_imgs_1.to(device)
noisy_imgs_2 = noisy_imgs_2.to(device)
noisy_imgs = noisy_imgs.to(device)
clean_imgs = clean_imgs.to(device)

# ----------- AUGMENTATIONS TESTS: EVERYTHING SEEMS TO BE FINE ----------------------
# ----------------------- DELETE AFTERWARDS -----------------------------------------
"""
new_imgs_1, new_imgs_2 = data_augmentation(noisy_imgs_1[:1,:,:,:], noisy_imgs_2[:1,:,:,:])

print(new_imgs_1.shape)

for i in range(len(new_imgs_1)):
    cv2.imwrite(f'new_1_{i}.png', new_imgs_1[i].permute(1,2,0).cpu().numpy())
    cv2.imwrite(f'new_2_{i}.png', new_imgs_2[i].permute(1,2,0).cpu().numpy())
cv2.imwrite(f'noisy_1.png', noisy_imgs_1[0].permute(1,2,0).cpu().numpy())
cv2.imwrite(f'noisy_2.png', noisy_imgs_2[0].permute(1,2,0).cpu().numpy())
"""
# ----------------------------------------------------------------------------------

# ---------------------- FORWARD PASS TESTS; JUST TO MAKE SURE IT'S WORKING --------
# ----------------------------DELETE AFTERWARDS ------------------------------------
"""
model = Model()
print(model.forward(noisy_imgs_1[:10]).shape)
"""

# ------------------------- OUTPUT TEST --------------------------------------
# ----------------------- DELETE AFTERWARDS ----------------------------------

model = Model()
model.load_pretrained_model()
output = model.forward(noisy_imgs_1[:1])
cv2.imwrite(f'noisy_1.png', noisy_imgs_1[0].permute(1,2,0).cpu().numpy())
cv2.imwrite(f'noisy_2.png', noisy_imgs_2[0].permute(1,2,0).cpu().numpy())
cv2.imwrite(f'output.png', output[0].permute(1,2,0).cpu().detach().numpy())

# ------------------- CODE TO TRAIN --------------------------
"""
model = Model()
model.to(device)
model.train(noisy_imgs_1, noisy_imgs_2, use_augs = True)
torch.save(model.state_dict(), './test_model.pth')
"""