import torch



a=torch.tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
print(a)
print(a.shape)
a=torch.softmax(a,dim=-1)
print(a)