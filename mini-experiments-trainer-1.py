
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



n_epochs=20

val_prop = 0.4

batch_size=64
fractions_to_remove = [0.3]
folds = [1,2,3]
sequence_lengths = [99]
start = time.time()

methods = ['prune_convolutions_only','prune_neurons_only']

for method in methods:
    for fold_number in folds:
        for sequence_length in sequence_lengths:

            if method=='prune_convolutions_only':
                for fraction_to_remove in fractions_to_remove:        
                    print ( "Training fold %s with %s method using sequence length %s and removing %s percent of weights"%(fold_number, method, sequence_length, int(fraction_to_remove*100)))
                    # This one takes a lot of time!!
                    dir_name = "fold_%s_models"%(fold_number)
                    dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
                    dir_name = os.path.join(dir_name, 'unpruned_model')
                    pruned_models = [torch.load(os.path.join(dir_name,'%s.pth'%(appliance_name))) for appliance_name in appliances]
                    for pruned_model in pruned_models:
                        remove_filters_only(pruned_model, fraction_to_remove)
                        pruned_model.cuda()
                    percent_to_remove = int(fraction_to_remove*100)
                    
                    train_fold(pruned_models, 'only_convolutions_pruned_model_%s_percent'%(percent_to_remove), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)

            elif method=='prune_neurons_only':
                for fraction_to_remove in fractions_to_remove:        
                    print ( "Training fold %s with %s method using sequence length %s and removing %s percent of weights"%(fold_number, method, sequence_length, int(fraction_to_remove*100)))
                    # This one takes a lot of time!!
                    dir_name = "fold_%s_models"%(fold_number)
                    dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
                    dir_name = os.path.join(dir_name, 'unpruned_model')
                    pruned_models = [torch.load(os.path.join(dir_name,'%s.pth'%(appliance_name))) for appliance_name in appliances]
                    for pruned_model in pruned_models:
                        remove_neurons_only(pruned_model, fraction_to_remove)
                        pruned_model.cuda()
                    percent_to_remove = int(fraction_to_remove*100)
                    
                    train_fold(pruned_models, 'only_neurons_pruned_model_%s_percent'%(percent_to_remove), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)


end = time.time()

print ("Total script runtime: ",end-start)

