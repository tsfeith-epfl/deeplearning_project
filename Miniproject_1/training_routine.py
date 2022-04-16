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


experiment_name = ''

criterions = [nn.L1Loss, nn.MSELoss]
criterions_str = ['L1', 'MSE']

optimizers = [optim.SGD, optim.Adam]
optimizers_str = ['SGD', 'Adam']

augs_str = ['no_augs', 'augs']
local_blur_str = ['no_local_blur', 'local_blur']


for i in range(len(criterions)):
    for j in range(len(optimizers)):
        for use_augs in [0, 1]:
            if use_augs:
                for gauss_kernel_size in [1,3]:
                    for local_blur in [0, 1]:
                        experiment_name = f"{criterions_str[i]}_{optimizers_str[j]}_{augs_str[use_augs]}_gkernel{gauss_kernel_size}_{local_blur_str[local_blur]}"
                        if os.path.exists(f"./outputs/{experiment_name}"):
                            continue
                        print(f"Running {experiment_name}\n")
                        model = Model()
                        model.nb_epochs = 25
                        model.to(device)
                        model.criterion = criterions[i]()
                        model.optimizer = optimizers[j](model.parameters(), lr = 1e-4)
                        model.train(noisy_imgs_1,
                                    noisy_imgs_2,
                                    use_augs = use_augs,
                                    gauss_kernel_size = gauss_kernel_size,
                                    local_blur = local_blur)
                        
                        os.makedirs(f"./outputs/{experiment_name}")
                        torch.save(model.state_dict(), f'./outputs/{experiment_name}/bestmodel.pth')

                        pred = model.predict(noisy_imgs)
                        psnr_val = psnr(pred/255, clean_imgs/255)
                        val_loss = model.criterion(pred, clean_imgs)
                        f = open(f"./outputs/{experiment_name}/results.txt", "w")
                        f.write(f"PSNR: {psnr_val}\nVal loss: {val_loss}")
                        f.close()
                        output = model.forward(noisy_imgs_1[:1])
                        cv2.imwrite(f'./outputs/{experiment_name}/noisy_1.png', noisy_imgs_1[0].permute(1,2,0).cpu().numpy())
                        cv2.imwrite(f'./outputs/{experiment_name}/noisy_2.png', noisy_imgs_2[0].permute(1,2,0).cpu().numpy())
                        cv2.imwrite(f'./outputs/{experiment_name}/output.png', output[0].permute(1,2,0).cpu().detach().numpy())
            else:
                for epoch in [25, 125]:
                    experiment_name = f"{criterions_str[i]}_{optimizers_str[j]}_{augs_str[use_augs]}_{epoch}epochs"
                    if os.path.exists(f"./outputs/{experiment_name}"):
                        continue
                    print(f"Running {experiment_name}\n")
                    model = Model()
                    model.nb_epochs = epoch
                    model.to(device)
                    model.criterion = criterions[i]()
                    model.optimizer = optimizers[j](model.parameters(), lr = 1e-4)
                    model.train(noisy_imgs_1, noisy_imgs_2, use_augs = False)

                    # Create a new directory because it does not exist 
                    os.makedirs(f"./outputs/{experiment_name}")
                    torch.save(model.state_dict(), f'./outputs/{experiment_name}/bestmodel.pth')
                    pred = model.predict(noisy_imgs)
                    psnr_val = psnr(pred/255, clean_imgs/255)
                    val_loss = model.criterion(pred, clean_imgs)
                    f = open(f"./outputs/{experiment_name}/results.txt", "w")
                    f.write(f"PSNR: {psnr_val}\nVal loss: {val_loss}")
                    f.close()
                    output = model.forward(noisy_imgs_1[:1])
                    cv2.imwrite(f'./outputs/{experiment_name}/noisy_1.png', noisy_imgs_1[0].permute(1,2,0).cpu().numpy())
                    cv2.imwrite(f'./outputs/{experiment_name}/noisy_2.png', noisy_imgs_2[0].permute(1,2,0).cpu().numpy())
                    cv2.imwrite(f'./outputs/{experiment_name}/output.png', output[0].permute(1,2,0).cpu().detach().numpy())