import model
import torch

if __name__ == '__main__':
    
    # ReLU tests
    print('ReLU')
    
    relu_ours = model.ReLU()
    relu = torch.nn.functional.relu
    
    x = torch.normal(0, 1, (1000000,1)).to('cuda')
    y = relu(x)
    y_ours = relu_ours.forward(x)
    
    print(f'Our mean: {y_ours.abs().mean()}+-{y_ours.abs().std()}')
    print(f'PyTorch mean: {y.abs().mean()}+-{y.abs().std()}')
    print(f'Mean of the difference: {(y - y_ours).abs().mean()}+-{(y - y_ours).abs().std()}')
    
    # Sigmoid tests
    print('\nSIGMOID')
    
    sigmoid_ours = model.Sigmoid()
    sigmoid = torch.sigmoid
    
    x = torch.normal(0, 1, (1000000,1)).to('cuda')
    y = sigmoid(x)
    y_ours = sigmoid_ours.forward(x)
    
    print(f'Our mean: {y_ours.abs().mean()}+-{y_ours.abs().std()}')
    print(f'PyTorch mean: {y.abs().mean()}+-{y.abs().std()}')
    print(f'Mean of the difference: {(y - y_ours).abs().mean()}+-{(y - y_ours).abs().std()}')
    
    # MSE tests
    print('\nMSE')
    
    mse_ours = model.MSE()
    mse = torch.nn.functional.mse_loss
    
    x = torch.normal(0, 1, (1000000,1)).to('cuda')
    x1 = torch.normal(0, 1, (1000000,1)).to('cuda')
    y = mse(x, x1)
    y_ours = mse_ours.forward(x, x1)
    
    print(f'Our mean: {y_ours.abs().mean()}+-{y_ours.abs().std()}')
    print(f'PyTorch mean: {y.abs().mean()}+-{y.abs().std()}')
    print(f'Mean of the difference: {(y - y_ours).abs().mean()}+-{(y - y_ours).abs().std()}')
    
    # Conv tests
    print('\nCONV')
    
    conv_ours = model.Conv2d(3,3,(3,3),stride=2)
    conv = torch.nn.Conv2d(3,3,(3,3),stride=2).to(torch.float).to('cuda')
    
    x = torch.normal(0, 1, (1000000,3,10,10)).to('cuda')
    y = conv(x)
    y_ours = conv_ours.forward(x)
    
    print(f'Our mean: {y_ours.abs().mean()}+-{y_ours.abs().std()}')
    print(f'PyTorch mean: {y.abs().mean()}+-{y.abs().std()}')
    print(f'Mean of the difference: {(y - y_ours).abs().mean()}+-{(y - y_ours).abs().mean((1,2,3)).std()}')
    

    # Upsample tests
    print('\nUPSAMPLE')
    
    upsample_ours = model.Upsampling(2, 3, 3)
    upsample = torch.nn.functional.upsample
    
    x = torch.normal(0, 1, (10000,3,10,10)).to('cuda')
    y = upsample(x, 20)
    y_ours = upsample_ours.forward(x)
    
    print(f'Our mean: {y_ours.abs().mean()}+-{y_ours.abs().std()}')
    print(f'PyTorch mean: {y.abs().mean()}+-{y.abs().std()}')
    print(f'Mean of the difference: {(y - y_ours).abs().mean()}+-{(y - y_ours).abs().mean((1,2,3)).std()}')