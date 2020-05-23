import torch

# t = torch.ones(2,1,2,1)
# print(t)
# print(t.shape)
# r = torch.squeeze(t)
# print(r)
# print(r.shape)
# s = torch.squeeze(r)
# print(s)
# print(s.shape)

# x = torch.Tensor([1, 2, 3])
# print(x)
# r = torch.unsqueeze(x, 0)       # Size: 1x3
# print(r)
# r = torch.unsqueeze(x, 1) 
# print(r) 

input = torch.randn(3, 5, requires_grad=True)
print(input)
target = torch.randint(5, (3,), dtype=torch.int64)
print(target)