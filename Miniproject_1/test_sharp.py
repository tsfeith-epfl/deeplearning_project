import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import glob
from model import Model as Model_TConv
from model_upsample import Model as Model_Upsample
from model import psnr
import cv2

if __name__ == '__main__':
    noisy_imgs, clean_imgs = torch.load('val_data.pkl')
    noisy_imgs = noisy_imgs.to(torch.float)
    clean_imgs = clean_imgs.to(torch.float)    
        
    file = "outputs_TConv/MSE_Adam_yes_sharp_no_crops_60epochs/bestmodel.pth"
    for sharp_factor in [1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95, 20]:
        model = Model_TConv()
        model.criterion = nn.MSELoss()
        model.load_state_dict(torch.load(file, map_location='cpu'))

        model.to('cpu')
        pred = model.predict(noisy_imgs, sharpen = True, sharpen_factor = sharp_factor)

        psnr_val = psnr(pred/255, clean_imgs/255)
        val_loss = model.criterion(pred, clean_imgs)

        print(f'Sharp Factor {sharp_factor} -> PSNR = {psnr_val}; Val Loss = {val_loss}')