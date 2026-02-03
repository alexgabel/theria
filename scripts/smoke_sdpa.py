import torch
from theria import sdpa

q = torch.randn(2, 1, 8, 16, requires_grad=True)
k = torch.randn(2, 1, 8, 16, requires_grad=True)
v = torch.randn(2, 1, 8, 16, requires_grad=True)

out = sdpa(q, k, v)
loss = out.sum()
loss.backward()

print("OK")