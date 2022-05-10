from model import Model
from model import psnr
import os
import torch
from torch import nn
from torch import optim
import cv2
from time import perf_counter

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


experiment_name = ''

gauss_strength = [1., 5., 10.]
starting_lr = [5e-3, 1e-3, 5e-4, 1e-4]
lr_decrease = [1, 2, 3]


for g_str in gauss_strength:
    for dec in lr_decrease:
        for lr in starting_lr:
            experiment_name = f"finetuning/{g_str}gauss_{lr}lr_{dec}dec"
            if os.path.exists(f"./outputs_{experiment_name}"):
                continue
            print(f"\nRunning {experiment_name}...")
            model = Model()
            model.to(device)
            model.criterion = nn.MSELoss()
            model.optimizer = optim.Adam(model.parameters(), lr = lr)
            start = perf_counter()
            model.train(noisy_imgs_1,
                        noisy_imgs_2,
                        20,
                        sharpen = False,
                        use_crops = False,
                        gauss_str = g_str,
                        scheduler_step = dec)
            end = perf_counter()
            os.makedirs(f"./outputs_{experiment_name}")
            torch.save(model.state_dict(), f"./outputs_{experiment_name}/bestmodel.pth")
            model.to('cpu')
            pred = model.predict(noisy_imgs, sharpen = False)
            psnr_val = psnr(pred/255, clean_imgs/255)
            val_loss = model.criterion(pred, clean_imgs)
            out_dir = f"./outputs_{experiment_name}/"
            f = open(f"./{out_dir}results.txt", "w")
            f.write(f"PSNR: {psnr_val}\nVal loss: {val_loss}")
            print(psnr_val, val_loss, (end-start)/60)
            f.close()
            output = model.forward(noisy_imgs[3:4])
            cv2.imwrite(f'./{out_dir}noisy_1.png', noisy_imgs[3].permute(1,2,0).numpy())
            cv2.imwrite(f'./{out_dir}noisy_2.png', clean_imgs[3].permute(1,2,0).numpy())
            cv2.imwrite(f'./{out_dir}output.png', output[0].permute(1,2,0).detach().numpy())