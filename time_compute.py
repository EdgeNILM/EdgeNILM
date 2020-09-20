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
sequence_lengths = [499, 99]
cuda=False

results_arr = []

for method in ['fully_shared_mtl_iterative_pruning','fully_shared_mtl_pruning','fully_shared_mtl','unpruned_model','tensor_decomposition','normal_pruning','iterative_pruning']:
    
        print ("Batch size:", batch_size)
        for sequence_length in sequence_lengths:

            if method=='unpruned_model':
                print ("-"*50)
                print ("Results unpruned model; sequence length: %s "%(sequence_length))
                compute_model_stats('unpruned_model', appliances, fold_number, sequence_length, batch_size, results_arr)
                
                print ("-"*50)
                print ("\n\n\n")
            
            elif method=='normal_pruning':
                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for %s percent Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "pruned_model_%s_percent" %(pruned_percentage)
                    compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)
                    
                    print ("-"*50)
                    print ("\n\n\n")


            elif method=='iterative_pruning':        

                for iterative_pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for %s percent Iterative Pruning; sequence length: %s "%(iterative_pruned_percentage, sequence_length))
                    model_name = "iterative_model_%s_percent" %(iterative_pruned_percentage)
                    compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)
                    
                    print ("-"*50)
                    print ("\n\n\n")

            elif method=='tensor_decomposition':

                for rank in [1,2, 4,8 ]:
                    print ("-"*50)
                    print ("Results for rank %s tensor decomposition; sequence length: %s "%(rank, sequence_length))
                    model_name = 'tensor_decomposition_rank_%s'%(rank)
                    compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                    print ("-"*50)
                    print ("\n\n\n")
            elif method == 'fully_shared_mtl':
                print ("-"*50)
                print ("Results for Fully shared MTL Model; sequence length: %s "%(sequence_length))
                model_name = method
                compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                print ("-"*50)
                print ("\n\n\n")
        
            elif method == 'normal_mtl':
                print ("-"*50)
                print ("Results for Normal  MTL Model; sequence length: %s "%(sequence_length))
                model_name = method
                compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                print ("-"*50)
                print ("\n\n\n")

            
            elif method=='fully_shared_mtl_pruning':

                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for Fully shared MTL %s percent Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "fully_shared_mtl_pruning_%s_percent" %(pruned_percentage)
                    compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)
                    
                    print ("-"*50)
                    print ("\n\n\n")
            
            
            elif method=='fully_shared_mtl_iterative_pruning':

                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for Fully shared MTL %s percent Iterative Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "fully_shared_mtl_iterative_model_%s_percent" %(pruned_percentage)
                    compute_model_stats(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                    
                    print ("-"*50)
                    print ("\n\n\n")


        
columns  = ['Model Name',"Sequence Length", "Time taken", "Model size"]

results_arr= np.array(results_arr)
df = pd.DataFrame(data=results_arr, columns=columns, index = range(len(results_arr)))
df.to_csv('inference-times-and-model-weights.csv',index=False)

