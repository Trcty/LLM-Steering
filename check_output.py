import torch 

result = torch.load('/scratch/zc1592/small_data/experiments/focal/bert-base-uncased_0.01_0.5_0.5_1.0_1/output.pt')
rounded = torch.round(result, decimals = 2)
print(rounded.shape)
non_zero_list = []
for i in range(rounded.shape[0]):
    non_zero_counts = (rounded[i,:] != 0).sum()
    non_zero_list.append(non_zero_counts.item())
print(non_zero_list)