import numpy as np
import pandas as pd
import time
import    math
import matplotlib.pyplot as plt
import os

import math
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import time
import sys


from functions import *
from networks import Seq2Point
from pthflops import count_ops

from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"]=""

class sillymodel(nn.Module):

    def __init__(self):
        super(sillymodel, self).__init__()
        self.conv1 = torch.nn.Conv1d(out_channels = 30 , kernel_size=10, in_channels = 1)
    
    def forward(self, X):
        x = self.conv1(X)
        x = F.relu(x)

        return x

sequence_length = 99


model = Seq2Point(99,False)#sillymodel()
model.to('cpu')

print (summary(model, (1, sequence_length)))

flops_tensor = torch.rand(1,1,sequence_length).to('cpu')

total_flops=count_ops(model, flops_tensor)[0]
