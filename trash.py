import torch

a = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[11, 22, 33], [44, 55, 66], [77, 88, 99]]])
b = torch.flip(a, [1, 2])
print(a)
print(b)

