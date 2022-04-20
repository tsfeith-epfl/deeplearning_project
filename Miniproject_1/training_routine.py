from model import Model as Model_TConv
from model_upsample import Model as Model_Upsample
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

models = [Model_TConv, Model_Upsample]
models_str = ['TConv', 'Upsample']

criterions = [nn.L1Loss, nn.MSELoss]
criterions_str = ['L1', 'MSE']

sharp_str = ['no_sharp', 'yes_sharp']
crop_str = ['no_crops', 'yes_crops']

optimizers = [optim.SGD, optim.Adam]
optimizers_str = ['SGD', 'Adam']
lr_optim = [1e-5, 1e-4]

lr_crit = [0.5, 0.05]


for i in range(len(models)):
    for j in range(len(criterions)):
        for k in range(len(optimizers)):
            for sharpen in [0, 1]:
                for crop in [0, 1]:
                    if crop:
                        experiment_name = f"{models_str[i]}/{criterions_str[j]}_{optimizers_str[k]}_{sharp_str[sharpen]}_{crop_str[crop]}_20epochs"
                        if os.path.exists(f"./outputs_{experiment_name}"):
                            continue
                        print(f"\nRunning {experiment_name}...")
                        model = models[i]()
                        model.nb_epochs = 20
                        model.to(device)
                        model.criterion = criterions[j]()
                        model.optimizer = optimizers[k](model.parameters(), lr = lr_optim[k] * lr_crit[j])
                        model.train(noisy_imgs_1,
                                    noisy_imgs_2,
                                    use_SSIM = criterions_str[j] == 'SSIM',
                                    sharpen = sharpen,
                                    use_crops = crop)
                        
                        os.makedirs(f"./outputs_{experiment_name}")
                        torch.save(model.state_dict(), f"./outputs_{experiment_name}/bestmodel.pth")
                    else:
                        for epochs in [20, 60]:
                            experiment_name = f"{models_str[i]}/{criterions_str[j]}_{optimizers_str[k]}_{sharp_str[sharpen]}_{crop_str[crop]}_{epochs}epochs"
                            if os.path.exists(f"./outputs_{experiment_name}"):
                                continue
                            print(f"\nRunning {experiment_name}...")
                            model = models[i]()
                            model.nb_epochs = epochs
                            model.to(device)
                            model.criterion = criterions[j]()
                            model.optimizer = optimizers[k](model.parameters(), lr = lr_optim[k] * lr_crit[j])
                            model.train(noisy_imgs_1,
                                        noisy_imgs_2,
                                        use_SSIM = criterions_str[j] == 'SSIM',
                                        sharpen = sharpen,
                                        use_crops = crop)

                            os.makedirs(f"./outputs_{experiment_name}")
                            torch.save(model.state_dict(), f"./outputs_{experiment_name}/bestmodel.pth")