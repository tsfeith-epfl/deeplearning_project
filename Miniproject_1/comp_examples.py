import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import glob
from model import Model as Model_TConv
from model_upsample import Model as Model_Upsample
from model import psnr
import cv2


if __name__ == '__main__':
    
    noisy_imgs_1, noisy_imgs_2 = torch.load('train_data.pkl')
    noisy_imgs, clean_imgs = torch.load('val_data.pkl')
    noisy_imgs_1 = noisy_imgs_1.to(torch.float)
    noisy_imgs_2 = noisy_imgs_2.to(torch.float)
    noisy_imgs = noisy_imgs.to(torch.float)
    clean_imgs = clean_imgs.to(torch.float)
    
    for index, file in enumerate(glob.glob("outputs*/*/*.pth")):
        model_type = file.split("/")[0][8:]
        params = file.split("/")[1].split("_")
        out_dir = file[:-13]
        
        criterion = params[0]
        sharpen = params[2]
    
        if model_type == 'TConv':
            model = Model_TConv()
        else:
            model = Model_Upsample()

        
        if criterion == 'L1':
            model.criterion = nn.L1Loss()
        elif criterion == 'MSE':
            model.criterion = nn.MSELoss()

        model.load_state_dict(torch.load(file, map_location='cpu'))

        model.to('cpu')
        pred = model.predict(noisy_imgs, sharpen = sharpen == 'yes')
        # pred = model.predict(noisy_imgs)
        
        psnr_val = psnr(pred/255, clean_imgs/255)
        val_loss = model.criterion(pred, clean_imgs)
        f = open(f"./{out_dir}results.txt", "w")
        f.write(f"PSNR: {psnr_val}\nVal loss: {val_loss}")
        f.close()
        output = model.forward(noisy_imgs[3:4])
        cv2.imwrite(f'./{out_dir}noisy_1.png', noisy_imgs[3].permute(1,2,0).numpy())
        cv2.imwrite(f'./{out_dir}noisy_2.png', clean_imgs[3].permute(1,2,0).numpy())
        cv2.imwrite(f'./{out_dir}output.png', output[0].permute(1,2,0).detach().numpy())
        
        print(f"{file}: {psnr_val}, {val_loss}")
        print(f"{index + 1}/{len(glob.glob('outputs*/*/*.pth'))} done!\n") 
