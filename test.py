import math

import torch
import torch.nn as nn
# a = torch.randn([64,1,200])
# a = a.reshape(a.shape[0],-1)
# print(a[32:60].shape)
# linear = nn.Linear(in_features=100,out_features=50)
# b = linear(a)
# out = nn.functional.softmax(b,dim = -1)
# print(out.shape)
pooling = nn.AvgPool1d(kernel_size=2)
a = torch.randn([100,2])
print(a)
b = nn.functional.softmax(a,dim=-1)
print(b)
# print(pooling(a).shape)
print(b.shape)
loss_fn = nn.CrossEntropyLoss()
# 方便理解，此处假设batch_size = 1
# x_input = torch.tensor([1,0,1,1,0,0,0,1])   # 预测2个对象，每个对象分别属于三个类别分别的概率
# # 需要的GT格式为(2)的tensor,其中的值范围必须在0-2(0<value<C-1)之间。
# x_target = torch.tensor([1,1,1,1,1,1,1,1])  # 这里给出两个对象所属的类别标签即可，此处的意思为第一个对象属于第0类，第二个我对象属于第2类
# loss = loss_fn(x_input, x_target)
# print('loss:\n', loss)

a = torch.randn([100,100])
# a = nn.functional.relu(a)
# print(a)
b = torch.randn([100,1])
c = torch.randn([100,1])
d = torch.mul(torch.mul(a,b),c)
print(d.shape)
# c = torch.div(a,b)
# d = torch.exp(c)
# print(d.shape)
# # print(torch.mul(a,b).shape)
# f = torch.tensor([[1,6],
#                   [2,-1],
#                   [2,5]])
# print(f.shape)
# index = torch.argmax(f,dim=-1).unsqueeze(-1)
# index = index.expand(index.shape[0],100).unsqueeze(-1)
# print(index.shape)
# a = torch.tensor([0]).repeat(10)
# print(a)

out = torch.tensor([[ 1,  2, 10000],
        [ 4,  5,  6]])

out = nn.functional.normalize(out.float(), p=1, dim=1)
print(out)
