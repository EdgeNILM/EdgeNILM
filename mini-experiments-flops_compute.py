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

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from functions import *
from networks import Seq2Point
from pthflops import count_ops



def compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr, num_samples=5):

    dir_name = "fold_%s_models"%(fold_number)
    dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
    dir_name = os.path.join(dir_name, model_name)
    print (dir_name)
    models_dir = dir_name

    parameters_path = os.path.join(dir_name, 'parameters.json')
    f = open(parameters_path)
    parameters = json.load(f)

    all_appliances_predictions = []

    total_flops = 0
    set_seed()

    flops_tensor = torch.rand(1,1,sequence_length).to('cpu')

    if 'mtl' in model_name:
        model_path = os.path.join(dir_name, "weights.pth")
        model = torch.load(model_path,map_location=torch.device('cpu'))    
        model.eval()    
        total_flops+=count_ops(model, flops_tensor)[0]

    else:

        for appliance_index, appliance_name in enumerate(appliances):
            
            model_path = os.path.join(dir_name, "%s.pth"%(appliance_name))
            model = torch.load(model_path,map_location=torch.device('cpu'))                
            model.eval()    
            total_flops+=count_ops(model, flops_tensor)[0]


    print (total_flops)

    results = []
    results.append(model_name)
    results.append(sequence_length)
    results.append(total_flops)
    results_arr.append(results)


appliances = ["fridge",'dish washer','washing machine']
appliances.sort()

batch_size=1
fold_number=1 # The fold weights to compute the run time of NN
sequence_lengths = [ 99]
cuda=False

results_arr = []

for method in ['only_convolutions_pruned_model_30_percent','only_neurons_pruned_model_30_percent','not_fully_shared_mtl']:
    
        print ("Batch size:", batch_size)
        for sequence_length in sequence_lengths:
                        
                print ("-"*50)
                print ("Results %s model; sequence length: %s "%(method, sequence_length))
                compute_model_stats(method, appliances, fold_number, sequence_length, batch_size, results_arr)
                
                print ("-"*50)
                print ("\n\n\n")


           

        
columns  = ['Model Name',"Sequence Length", 'Total Flops']

results_arr= np.array(results_arr)
df = pd.DataFrame(data=results_arr, columns=columns, index = range(len(results_arr)))
df.to_csv('mini-experiments-flops.csv',index=False)

