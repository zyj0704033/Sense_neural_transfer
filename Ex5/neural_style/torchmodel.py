
import torch
import torch.nn as nn
import torch.legacy.nn as lnn

from functools import reduce
from torch.autograd import Variable

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class LambdaMap(LambdaBase):
    def forward(self, input):
        return list(map(self.lambda_func,self.forward_prepare(input)))

class LambdaReduce(LambdaBase):
    def forward(self, input):
        return reduce(self.lambda_func,self.forward_prepare(input))


torchmodel = nn.Sequential( # Sequential,
	nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True),
	nn.Dropout(0.5),
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(16384,256)), # Linear,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(256,256)), # Linear,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(256,212)), # Linear,
)