from model import Model
from model import psnr
import os
import torch
from torch import nn
from torch import optim
import cv2

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
noisy_imgs = noisy_imgs.to(device)
clean_imgs = clean_imgs.to(device)
print('DATA PASSED TO DEVICE')

exps = [[Model, nn.MSELoss, optim.Adam, False, False],
        [Model, nn.MSELoss, optim.Adam, True, False],
        [Model, nn.MSELoss, optim.SGD, True, False]]

exps_name = [['TConv', 'MSE', 'Adam', 'no_sharp', 'no_crops'],
             ['TConv', 'MSE', 'Adam', 'yes_sharp', 'no_crops'],
             ['TConv', 'MSE', 'SGD', 'yes_sharp', 'no_crops']]

for index, exp in enumerate(exps):
    experiment_name = f"{exps_name[index][1]}_{exps_name[index][2]}_{exps_name[index][3]}_{exps_name[index][4]}"
    if os.path.exists(f"./psnr_time/{experiment_name}.txt"):
        continue
    print(f"\nRunning {experiment_name}...")
    model = exp[0]()
    model.to(device)
    model.criterion = exp[1]()
    model.optimizer = exp[2](model.parameters(), lr = 5e-4 if exps_name[index][2] == 'Adam' else 5e-6)
    psnr_vals = model.train(noisy_imgs_1,
                            noisy_imgs_2,
                            200,
                            sharpen = exp[3],
                            use_crops = exp[4],
                            test_input = noisy_imgs,
                            test_target = clean_imgs)
    f = open(f"./psnr_time/{experiment_name}.txt", 'w')
    vals_str = ""
    for val in psnr_vals:
        vals_str += f"{val}\n"
    f.write(vals_str)
    f.close()
