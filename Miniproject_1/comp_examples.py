import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import glob
from model import Model as Model_TConv
from model_upsample import Model as Model_Upsample
from model import psnr
import cv2

if __name__ == '__main__':
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    noisy_imgs, clean_imgs = torch.load('val_data.pkl')
    noisy_imgs = noisy_imgs.to(torch.float)
    clean_imgs = clean_imgs.to(torch.float)
    noisy_imgs = noisy_imgs.to(device)
    clean_imgs = clean_imgs.to(device)
    
    with torch.no_grad():
        for index, file in enumerate(glob.glob("outputs_finetuning/*/*.pth")):
            model_type = file.split("/")[0][8:]
            params = file.split("/")[1].split("_")
            out_dir = file[:-13]

            criterion = params[0]
            sharpen = params[2]

            if model_type == 'Upsample':
                model = Model_Upsample()
            else:
                model = Model_TConv()


            if criterion == 'L1':
                model.criterion = nn.L1Loss()
            elif criterion == 'MSE':
                model.criterion = nn.MSELoss()

            model.load_state_dict(torch.load(file, map_location='cpu'))

            model.to(device)
            psnr_vals = []
            for _ in range(100):
                samples = torch.rand(500)*noisy_imgs.shape[0]
                samples = samples.type(torch.int)
                samples = samples.to(device)
                pred = model.predict(torch.index_select(noisy_imgs, 0, samples))
                psnr_vals.append(psnr(pred/255, torch.index_select(clean_imgs, 0, samples)/255))
            mean_psnr = torch.mean(torch.Tensor(psnr_vals))
            std_psnr  = torch.std(torch.Tensor(psnr_vals))

            # pred = model.predict(noisy_imgs, sharpen = sharpen == 'yes')
            # pred = model.predict(noisy_imgs)

            # psnr_val = psnr(pred/255, clean_imgs/255)
            # val_loss = model.criterion(pred, clean_imgs)
            f = open(f"./{out_dir}results.txt", "w")
            f.write(f"PSNR: {mean_psnr}+-{std_psnr}")
            f.close()
            output = model.forward(noisy_imgs[3:4])
            cv2.imwrite(f'./{out_dir}noisy_1.png', noisy_imgs[3].cpu().permute(1,2,0).numpy())
            cv2.imwrite(f'./{out_dir}noisy_2.png', clean_imgs[3].cpu().permute(1,2,0).numpy())
            cv2.imwrite(f'./{out_dir}output.png', output[0].cpu().permute(1,2,0).detach().numpy())

            print(f"{file}: {mean_psnr}, {std_psnr}")
            print(f"{index + 1}/{len(glob.glob('outputs*/*/*.pth'))} done!\n") 
