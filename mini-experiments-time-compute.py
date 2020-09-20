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



def compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr, num_samples=500):

    dir_name = "fold_%s_models"%(fold_number)
    dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
    dir_name = os.path.join(dir_name, model_name)
    print (dir_name)
    models_dir = dir_name

    parameters_path = os.path.join(dir_name, 'parameters.json')
    f = open(parameters_path)
    parameters = json.load(f)

    all_appliances_predictions = []
    total_time = 0
    model_size = 0
    set_seed()
    batch_tensor = torch.from_numpy(np.random.random((batch_size, 1, sequence_length ))).float()

    total_time_taken = 0

    flops_tensor = torch.rand(1,1,sequence_length).to('cpu')

    if 'mtl' in model_name:
        model_path = os.path.join(dir_name, "weights.pth")
        model = torch.load(model_path,map_location=torch.device('cpu'))    
        model_size+= (int(os.stat(model_path).st_size)/(1024*1024))
        model.eval()    
        a = time.time()
        for i in range(num_samples//batch_size):
            prediction = model(batch_tensor)
        b = time.time()
        
        total_time_taken+=b-a

    else:

        for appliance_index, appliance_name in enumerate(appliances):
            
            model_path = os.path.join(dir_name, "%s.pth"%(appliance_name))
            model = torch.load(model_path,map_location=torch.device('cpu'))                
            model.eval()    
            a = time.time()
            for i in range(num_samples//batch_size):
                prediction = model(batch_tensor)
            b = time.time()
            total_time_taken+=b-a
            model_size+= (int(os.stat(model_path).st_size)/(1024*1024))
            


    results = []
    results.append(model_name)
    results.append(sequence_length)
    results.append((total_time_taken)*1000/num_samples)
    results.append(model_size)
    results_arr.append(results)


appliances = ["fridge",'dish washer','washing machine']
appliances.sort()

batch_size=1
fold_number=1 # The fold weights to compute the run time of NN
sequence_lengths = [99]
cuda=False

results_arr = []

for method in ['only_convolutions_pruned_model_30_percent','only_neurons_pruned_model_30_percent','not_fully_shared_mtl']:
    
        print ("Batch size:", batch_size)
        for sequence_length in sequence_lengths:
               
                print ("-"*50)
                print ("Results for Normal  %s; sequence length: %s "%(method,sequence_length))
                model_name = method
                compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                print ("-"*50)
                print ("\n\n\n")
            


        
columns  = ['Model Name',"Sequence Length", "Time taken", "Model size"]

results_arr= np.array(results_arr)
df = pd.DataFrame(data=results_arr, columns=columns, index = range(len(results_arr)))
df.to_csv('mini-experiments-times-and-model-weights.csv',index=False)

