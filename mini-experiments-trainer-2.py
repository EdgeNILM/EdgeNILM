
import numpy as np
import pandas as pd
import time
import  math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
from sklearn.metrics import  mean_absolute_error
import math
import sys
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import time


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from functions import *
from networks import *



cuda=True

appliances =  ["washing machine","fridge", "dish washer"]



n_epochs=60

val_prop = 0.4

batch_size=64
folds = [1,2,3]
sequence_lengths = [99]

start = time.time()

methods = ['not_fully_shared_mtl']

for method in methods:
    for fold_number in folds:
        for sequence_length in sequence_lengths:

                if method=='not_fully_shared_mtl':
                    print ( "Training fold %s with %s method using sequence length %s"%(fold_number, method, sequence_length))          
                    
                    mtl_model = [NotFullySharedMTL(sequence_length, len(appliances), cuda)]
                    train_fold(mtl_model, method, appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)



end = time.time()

print ("Total script runtime: ",end-start)

