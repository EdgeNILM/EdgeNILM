
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

method=sys.argv[1]

if 'mtl' in method:
  n_epochs=60
else:
  n_epochs=20


val_prop = 0.4

batch_size=64
fractions_to_remove = [0.3,0.6,0.9]
ranks = [1,2,4,8]
folds = [1,2,3]
sequence_lengths = [99, 499]
iterative_increment = 0.1
start = time.time()

for fold_number in folds:
  for sequence_length in sequence_lengths:
    
    if method=='unpruned_model':
      print ( "Training fold %s with %s method using sequence length %s"%(fold_number, 'unpruned_model', sequence_length))    
      """Unpruned Model"""
      unpruned_models = [Seq2Point(sequence_length,cuda) for i in range(len(appliances))]
      train_fold(unpruned_models, 'unpruned_model', appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)
      """Pruned Model"""

    elif method=='normal_pruning':
      for fraction_to_remove in fractions_to_remove:        
        print ( "Training fold %s with %s method using sequence length %s and removing %s percent of weights"%(fold_number, 'pruned', sequence_length, int(fraction_to_remove*100)))
        # This one takes a lot of time!!
        dir_name = "fold_%s_models"%(fold_number)
        dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
        dir_name = os.path.join(dir_name, 'unpruned_model')
        pruned_models = [torch.load(os.path.join(dir_name,'%s.pth'%(appliance_name))) for appliance_name in appliances]
        for pruned_model in pruned_models:
          remove_filters_and_neurons(pruned_model, fraction_to_remove)
          pruned_model.cuda()
        percent_to_remove = int(fraction_to_remove*100)
        
        train_fold(pruned_models, 'pruned_model_%s_percent'%(percent_to_remove), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)


    elif method=='iterative_pruning':
      layers = ['conv1','conv2','conv3','conv4','conv5']
      for increment in range(1,int(max(fractions_to_remove)/iterative_increment) + 1):
        print ( "Training fold %s with %s method using sequence length %s"%(fold_number, 'iterative_%s_percent'%(increment*10) , sequence_length))    

        dir_name = "fold_%s_models"%(fold_number)
        dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
        if increment==1:
          dir_name = os.path.join(dir_name, 'unpruned_model')
        else:
          dir_name = os.path.join(dir_name, 'iterative_model_%s_percent'%((increment-1)*10))

        pruned_models = [torch.load(os.path.join(dir_name,'%s.pth'%(appliance_name))) for appliance_name in appliances]

        if increment==1:
          model = pruned_models[0]
          num_convolution_filters = [getattr( model, layer).weight.shape[0] for layer in layers]
          num_dense_neurons = model.fc1.weight.shape[0]
          num_convolution_filters_to_remove = [int(n_filter * iterative_increment) for n_filter in num_convolution_filters]
          num_dense_neurons_to_remove = int(num_dense_neurons * iterative_increment)

        for pruned_model in pruned_models:
          iteratively_remove(pruned_model, num_convolution_filters_to_remove, num_dense_neurons_to_remove)
          pruned_model.cuda()
        train_fold(pruned_models, 'iterative_model_%s_percent'%(increment*10), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)


    elif method=='tensor_decomposition':
      for rank in ranks:
          """Tensor Decomposition"""        
          print ( "Training fold %s with %s method using sequence length %s using rank %s"%(fold_number, 'tensor', sequence_length, rank))
          dir_name = "fold_%s_models"%(fold_number)
          dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
          dir_name = os.path.join(dir_name, 'unpruned_model')
          td_models = [torch.load(os.path.join(dir_name,'%s.pth'%(appliance_name))) for appliance_name in appliances]
          for tensor_decomposition_model in td_models:
            cp_decompose_model(tensor_decomposition_model, rank)
            tensor_decomposition_model.cuda()
          train_fold(td_models, 'tensor_decomposition_rank_%s'%(rank), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)
    
    elif method=='fully_shared_mtl':
      print ( "Training fold %s with %s method using sequence length %s"%(fold_number, 'multi task learning model', sequence_length))          
      
      mtl_model = [FullySharedMTL(sequence_length, len(appliances), cuda)]
      train_fold(mtl_model, method, appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)

    elif method=='fully_shared_mtl_pruning':
      for fraction_to_remove in fractions_to_remove:        
        print ( "Training fold %s with %s method using sequence length %s and removing %s percent of weights"%(fold_number, 'Fully Shared MTL Pruning', sequence_length, int(fraction_to_remove*100)))
        # This one takes a lot of time!!
        dir_name = "fold_%s_models"%(fold_number)
        dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
        dir_name = os.path.join(dir_name, 'fully_shared_mtl')
        pruned_models = [torch.load(os.path.join(dir_name,'weights.pth'))]
        for pruned_model in pruned_models:
          remove_filters_and_neurons(pruned_model, fraction_to_remove)
          pruned_model.cuda()
        percent_to_remove = int(fraction_to_remove*100)        
        train_fold(pruned_models, 'fully_shared_mtl_pruning_%s_percent'%(percent_to_remove), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)


    elif method=='fully_shared_mtl_iterative_pruning':
      layers = ['conv1','conv2','conv3','conv4','conv5']
      for increment in range(1,int(max(fractions_to_remove)/iterative_increment) + 1):
        print ( "Training fold %s with %s method using sequence length %s"%(fold_number, 'mtl_iterative_%s_percent'%(increment*10) , sequence_length))    

        dir_name = "fold_%s_models"%(fold_number)
        dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
        if increment==1:
          dir_name = os.path.join(dir_name, 'fully_shared_mtl')
        else:
          dir_name = os.path.join(dir_name, 'fully_shared_mtl_iterative_model_%s_percent'%((increment-1)*10))

        pruned_models = [torch.load(os.path.join(dir_name,'weights.pth'))]

        if increment==1:
          model = pruned_models[0]
          num_convolution_filters = [getattr( model, layer).weight.shape[0] for layer in layers]
          num_dense_neurons = model.fc1.weight.shape[0]
          num_convolution_filters_to_remove = [int(n_filter * iterative_increment) for n_filter in num_convolution_filters]
          num_dense_neurons_to_remove = int(num_dense_neurons * iterative_increment)

        for pruned_model in pruned_models:
          iteratively_remove(pruned_model, num_convolution_filters_to_remove, num_dense_neurons_to_remove)
          pruned_model.cuda()
        train_fold(pruned_models, 'fully_shared_mtl_iterative_model_%s_percent'%(increment*10), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)




    # elif method=='global_pruning':
    #   for fraction_to_remove in fractions_to_remove:        
    #     print ( "Training fold %s with %s method using sequence length %s and removing %s percent of weights"%(fold_number, method , sequence_length, int(fraction_to_remove*100)))
    #     # This one takes a lot of time!!
    #     dir_name = "fold_%s_models"%(fold_number)
    #     dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
    #     dir_name = os.path.join(dir_name, 'unpruned_model')
    #     pruned_models = [torch.load(os.path.join(dir_name,'%s.pth'%(appliance_name))) for appliance_name in appliances]
    #     for pruned_model in pruned_models:
    #       global_pruning(pruned_model, fraction_to_remove)
    #       pruned_model.cuda()
    #     percent_to_remove = int(fraction_to_remove*100)
        
        # train_fold(pruned_models, 'global_pruned_model_%s_percent'%(percent_to_remove), appliances, fold_number, n_epochs, sequence_length, batch_size, 'adam', val_prop,num_of_minibatches_to_save_model=40)

end = time.time()

print ("Total script runtime: ",end-start)



