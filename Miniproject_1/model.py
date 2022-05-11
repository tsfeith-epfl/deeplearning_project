import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as F1
from torch.optim.lr_scheduler import StepLR
from torch import optim
from torchvision import transforms
import random
from time import perf_counter
import cv2

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
        self.optimizer = optim.Adam(self.parameters(), lr = 5e-5)

        # self.mini_batch_size = 625
        self.mini_batch_size = 125
        
        # Initialize weights
        self._init_weights()


    def _init_weights(self):
        """Initializes weights using He et al. (2015)."""
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight.data)
                module.bias.data.zero_()
                    
    def forward(self, x, sharpen = True, sharp_factor = 1.):
        
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

        x = F.relu(self.dec_conv1(x))
        
        if sharpen:
            x = F.relu(x + sharp_factor*(x - F1.gaussian_blur(x, 3)))

        return x

    def load_pretrained_model(self):
        """
        This loads the parameters saved in bestmodel.pth into the model.

        Returns
        -------
        None
        """
        best_state_dict = torch.load('bestmodel.pth', map_location='cpu')
        self.load_state_dict(best_state_dict)

    def train(self,
              train_input,
              train_target,
              epochs,
              n_local_crops = 2,
              sharpen = True,
              sharpen_factor = 1.,
              use_crops = True,
              test_input = None,
              test_target = None,
              gauss_str = 1.,
              scheduler_step = 2):
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
        psnr_vals = []
        n_samples = len(train_input)*(1 + n_local_crops) if use_crops else len(train_input) 
        print('\nTRAINING STARTING...')
        scheduler = StepLR(self.optimizer, step_size = epochs // scheduler_step)
        for e in range(epochs):
            epoch_loss = 0
            for b in range(0, train_input.size(0), self.mini_batch_size):
                self.optimizer.zero_grad()
                train = train_input.narrow(0, b, self.mini_batch_size)
                target = train_target.narrow(0, b, self.mini_batch_size)
                train, target = data_augmentations(train, target, use_crops = use_crops, n_local_crops = n_local_crops, gauss_str = gauss_str)
                output = self.forward(train, sharpen, sharpen_factor)
                loss = self.criterion(output, target)
                epoch_loss += loss.item()/n_samples
                loss.backward()
                self.optimizer.step()
            scheduler.step()
            print(scheduler.get_last_lr())
            if test_input != None:
                # print(psnr_vals)
                test_input = test_input.to('cpu')
                
                self.to('cpu')
                test_output = self.forward(test_input, sharpen, sharpen_factor)
                test_output = test_output.to('cuda')
                self.to('cuda')
                psnr_vals.append(psnr(test_output/255, test_target/255).item())
                print(psnr_vals[-1])
            print(f"Epoch {e+1}: Loss = {epoch_loss};")
            torch.cuda.empty_cache()
        return psnr_vals
        
    def predict(self, test_input, sharpen = True, sharpen_factor = 1.):
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
        return self.forward(test_input, sharpen, sharpen_factor)
        
    
def data_augmentations(imgs_1, 
                       imgs_2,
                       hflip_prob = 0.5,
                       vflip_prob = 0.5,
                       use_crops = True,
                       n_local_crops = 2,
                       gauss_str = 1.):
    H, W = imgs_1.shape[-2], imgs_1.shape[-1]
    
    if use_crops:
        global_augs = transforms.Compose([transforms.RandomHorizontalFlip(p=hflip_prob),
                                          transforms.RandomVerticalFlip(p=vflip_prob),
                                          transforms.RandomResizedCrop((H,W), scale = (0.7, 1.0)),
                                          AddGaussianNoise(0., gauss_str)
                                          ])

        local_augs = transforms.Compose([transforms.RandomHorizontalFlip(p=hflip_prob),
                                         transforms.RandomVerticalFlip(p=vflip_prob),
                                         transforms.RandomResizedCrop((H,W), scale = (0.05, 0.4)),
                                         AddGaussianNoise(0., gauss_str)
                                         ])
    else:
        global_augs = transforms.Compose([transforms.RandomHorizontalFlip(p=hflip_prob),
                                          transforms.RandomVerticalFlip(p=vflip_prob),
                                          AddGaussianNoise(0., gauss_str)
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

if __name__ == '__main__':

    noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl')
    noisy_imgs, clean_imgs = torch.load('val_data.pkl')
    noisy_imgs_1 = noisy_imgs_1.to(torch.float)
    noisy_imgs_2 = noisy_imgs_2.to(torch.float)
    noisy_imgs = noisy_imgs.to(torch.float)
    clean_imgs = clean_imgs.to(torch.float)
    print('DATA IMPORTED')


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('DEVICE DEFINED', device)
    noisy_imgs_1 = noisy_imgs_1.to(device)
    noisy_imgs_2 = noisy_imgs_2.to(device)
    noisy_imgs = noisy_imgs.to('cpu')
    clean_imgs = clean_imgs.to('cpu')
    print('DATA PASSED TO DEVICE')

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
    """
    model = Model()
    model.to(device)
    model.load_pretrained_model()
    output = model.forward(noisy_imgs_1[:1])
    cv2.imwrite(f'noisy_1.png', noisy_imgs_1[0].permute(1,2,0).cpu().numpy())
    cv2.imwrite(f'noisy_2.png', noisy_imgs_2[0].permute(1,2,0).cpu().numpy())
    cv2.imwrite(f'output.png', output[0].permute(1,2,0).cpu().detach().numpy())
    """
    # ---------------- CODE TO TRAIN --------------------
    """
    model = Model()
    model.load_pretrained_model()
    model.to(device)
    model.train(noisy_imgs_1, noisy_imgs_2, use_augs = False)
    # model.to('cuda')
    torch.save(model.state_dict(), './test_model.pth')
    """

    # --
    
    model = Model()
    model.to(device)
    model.criterion = nn.MSELoss()
    model.optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    start = perf_counter()
    model.train(noisy_imgs_1,
                noisy_imgs_2,
                20,
                sharpen = True,
                use_crops = True,
                gauss_str = 1,
                scheduler_step = 2)
    end = perf_counter()
    model.to('cpu')
    pred = model.predict(noisy_imgs, sharpen = False)
    psnr_val = psnr(pred/255, clean_imgs/255)
    val_loss = model.criterion(pred, clean_imgs)
    print(psnr_val, val_loss, (end-start)/60)
