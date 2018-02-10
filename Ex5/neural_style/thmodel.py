
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


thmodel = nn.Sequential( # Sequential,
	nn.Conv2d(3,32,(7, 7),(2, 2),(3, 3)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.Conv2d(32,32,(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,32,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(32,64,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.Conv2d(32,64,(1, 1),(2, 2)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,64,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(64,128,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.Conv2d(64,128,(1, 1),(2, 2)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,128,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(128,256,(3, 3),(2, 2),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.Conv2d(128,256,(1, 1),(2, 2)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	nn.Conv2d(256,256,(3, 3),(1, 1),(1, 1)),
	nn.ReLU(),
	Lambda(lambda x: x.view(x.size(0),-1)), # View,
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(65536,256)), # Linear,
	nn.ReLU(),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(256,256)), # Linear,
	nn.ReLU(),
	nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(256,212)), # Linear,
)