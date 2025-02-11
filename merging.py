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

combined = [] 
for i in range(25):
    print(i)
    combined.append(all_data[i][0][ipc * (i*4) : ipc * (i+1) * 4])
combined = torch.cat(combined, dim=0)
print(combined.shape)
label = all_data[0][1]

torch.save([combined, label], f"results/cifar100/conv3in_iterative_increase_grad_mse_nd2000_cut_niter2000_factor2_lr0.005_b_real128_mix_ipc2/interval_{interval_index}_data.pt")