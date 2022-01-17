from models.convnext import convnext_tiny
import torch
from torch import nn
from collections import OrderedDict

data = torch.randn(1, 3, 100, 100)
model = convnext_tiny()
res = model(data)

print(res.shape)

new_model = nn.Sequential(OrderedDict(list(model.named_children())[:-1]))
res = new_model(data)
print(res.shape)
