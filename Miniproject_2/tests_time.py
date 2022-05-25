from time import perf_counter
import model
import torch

if __name__ == '__main__':
    """
    # ReLU tests
    print('ReLU')
    
    relu_ours = model.ReLU()
    relu = torch.nn.functional.relu
    
    x = torch.Tensor([1.])
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = relu(x)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time PyTorch: {time_vals.mean()}+-{time_vals.std()}")
    
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = relu_ours.forward(x)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time Ours: {time_vals.mean()}+-{time_vals.std()}")
    
    # Sigmoid tests
    print('\nSIGMOID')
    
    sigmoid_ours = model.Sigmoid()
    sigmoid = torch.sigmoid
    
    x = torch.Tensor([1.])
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = sigmoid(x)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time PyTorch: {time_vals.mean()}+-{time_vals.std()}")
    
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = sigmoid_ours.forward(x)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time Ours: {time_vals.mean()}+-{time_vals.std()}")
    
    # MSE tests
    print('\nMSE')
    
    mse_ours = model.MSE()
    mse = torch.nn.functional.mse_loss
    
    x = torch.Tensor([1.])
    y = torch.Tensor([2.])
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = mse(x, y)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time PyTorch: {time_vals.mean()}+-{time_vals.std()}")
    
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = mse_ours.forward(x, y)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time Ours: {time_vals.mean()}+-{time_vals.std()}")
    # Conv tests
    print('\nCONV')
    
    conv_ours = model.Conv2d(3,3,(3,3),stride=2)
    conv = torch.nn.Conv2d(3,3,(3,3),stride=2).to(torch.float)
    
    x = torch.arange(75).view(1,3,5,5).to(torch.float)
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = conv.forward(x)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time PyTorch: {time_vals.mean()}+-{time_vals.std()}")
    
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = conv_ours.forward(x)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time Ours: {time_vals.mean()}+-{time_vals.std()}")
    """
    # Upsample tests
    print('\nUPSAMPLE')
    
    upsample_ours = model.Upsampling(2)
    upsample = torch.nn.functional.upsample
    
    x = torch.arange(75).view(1,3,5,5).to(torch.float)
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = upsample(x, scale_factor=2)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time PyTorch: {time_vals.mean()}+-{time_vals.std()}")
    
    time_vals = []
    for i in range(1000):
        start = perf_counter()
        for j in range(1000):
            y = upsample_ours.forward(x)
        end = perf_counter()
        time_vals.append((end - start)/1000)
    time_vals = torch.Tensor(time_vals)
    print(f"Time Ours: {time_vals.mean()}+-{time_vals.std()}")