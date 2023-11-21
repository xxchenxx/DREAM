import torch

increment = 4

interval_index = 1
ipc = 2 * (interval_index + 1)
all_data = []
for i in range(25):
    path = f"results/cifar100/conv3in_iterative_increase_grad_mse_nd2000_cut_niter2000_factor2_lr0.005_b_real128_mix_ipc2/interval_{interval_index}_data_{4*i}_{4*(i+1)}.pt"
    data = torch.load(path)
    all_data.append(data)

prev = f"results/cifar100/conv3in_iterative_increase_grad_mse_nd2000_cut_niter2000_factor2_lr0.005_b_real128_mix_ipc2/interval_{interval_index-1}_data.pt"
prev = torch.load(prev)
print(prev)
print(all_data[i][0])

for i in range(25):
    print(i)
    for j in range(100):
        print(j)
        print(prev[0][j*2+1] - all_data[i][0][j * 4+2])