import numpy as np
import pandas as pd
import time
import    math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
from sklearn.metrics import    mean_absolute_error, f1_score
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

def test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr):

    dir_name = "fold_%s_models"%(fold_number)
    dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
    dir_name = os.path.join(dir_name, model_name)

    models_dir = dir_name

    test_file_name = 'test_%s.h5'%(fold_number)
    all_appliances_mains_lst, all_appliances_truth = load_h5_file(test_file_name, appliances)
    
    # Taking the first 10% or 20%

    for i in range(len(all_appliances_mains_lst)):
        n = len(all_appliances_mains_lst[i])
        for j in range(n):
            df = all_appliances_mains_lst[i][j]
            all_appliances_mains_lst[i][j] = df.iloc[:int(fraction_to_test * len(df))]

            df = all_appliances_truth[i][j]
            all_appliances_truth[i][j] = df.iloc[:int(fraction_to_test * len(df))]

    if method!='zero':
        parameters_path = os.path.join(dir_name, 'parameters.json')
        f = open(parameters_path)
        parameters = json.load(f)

        all_appliances_predictions = []

        for appliance_index, appliance_name in enumerate(appliances):
            mains_mean = parameters[appliance_name]['mains_mean']
            mains_std = parameters[appliance_name]['mains_std']
            app_mean = parameters[appliance_name]['app_mean']
            app_std = parameters[appliance_name]['app_std']
            appliance_mains_dfs = all_appliances_mains_lst[appliance_index]

            no_of_homes = len(appliance_mains_dfs)
            if 'mtl' not in model_name:
                    model_path = os.path.join(dir_name, "%s.pth"%(appliance_name))
            else:
                    model_path = os.path.join(dir_name, "weights.pth")
            
            if not cuda:
                model = torch.load(model_path,map_location=torch.device('cpu'))    
            else:
                model = torch.load(model_path)    
            model.eval()    
            

            appliance_prediction = []

            for home_id in range(no_of_homes):
                home_mains = appliance_mains_dfs[home_id]
                l = len(home_mains)
                            
                processed_mains = mains_preprocessing([home_mains], sequence_length)
                processed_mains = (processed_mains - mains_mean)/mains_std
    
                
                if 'mtl' in model_name:
                    prediction = predict_mtl(model, processed_mains, appliance_index, cuda, batch_size)
                else:
                    prediction = predict(model, processed_mains, cuda, batch_size)
                
                prediction = prediction * app_std + app_mean
                prediction = prediction.flatten()
                prediction = np.where(prediction>0, prediction,0)
                df = pd.DataFrame({appliance_name: prediction})
                df.index = home_mains.index
                appliance_prediction.append(df)

                # print (home_mains.shape)
                # print ()
            all_appliances_predictions.append(appliance_prediction)


        
        # print ("Finished predicting for appliance %s"%(appliance_name))
    else:

        all_appliances_predictions = []

        for appliance_index, appliance_name in enumerate(appliances):
            
            appliance_mains_dfs = all_appliances_mains_lst[appliance_index]

            no_of_homes = len(appliance_mains_dfs)

            appliance_prediction = []

            for home_id in range(no_of_homes):
                home_mains = appliance_mains_dfs[home_id]
                l = len(home_mains)
                            
                prediction = np.zeros(l)
                df = pd.DataFrame({appliance_name: prediction})
                df.index = home_mains.index
                appliance_prediction.append(df)

                # print (home_mains.shape)
                # print ()
            all_appliances_predictions.append(appliance_prediction)

    results = []
    results.append(model_name)
    results.append(sequence_length)
    results.append(fold_number)
    results.append(batch_size)
    
    total_error = [0 for q in range(len(metrics))]

    for app_index, app_name in enumerate(appliances):
        
        for metric_index, metric in enumerate(metrics):
            if metric=='mae':
                truth_ = pd.concat(all_appliances_truth[app_index],axis=0).values
                pred_ = pd.concat(all_appliances_predictions[app_index],axis=0).values
                error = mean_absolute_error(truth_, pred_)
                
            elif metric=='f1-score':
                truth_ = pd.concat(all_appliances_truth[app_index],axis=0).values
                pred_ = pd.concat(all_appliances_predictions[app_index],axis=0).values
                truth_ = np.where(truth_>threshold, 1,0)
                pred_ = np.where(pred_>threshold, 1,0)
                error = f1_score(truth_, pred_)            
            else: 
                total_ground_truth_usage = [np.sum(df.values) for df in all_appliances_truth[app_index]]
                total_prediction_usage = [np.sum(df.values) for df in all_appliances_predictions[app_index]]
                total_mains_usage = [np.sum(df.values) for df in all_appliances_mains_lst[app_index]]

                energy_incorrectly_assigned = [ (total_ground_truth_usage[home]/total_mains_usage[home] )- (total_prediction_usage[home]/total_mains_usage[home]) for home in range(len(total_ground_truth_usage)) ]
                error = 100*np.mean(np.abs(energy_incorrectly_assigned))

            print ("%s %s Error: %s"%(app_name, metric,error))

            results.append(error)
            total_error[metric_index]+=error
        
        if plot:
            truth_ = pd.concat(all_appliances_truth[app_index],axis=0).values
            pred_ = pd.concat(all_appliances_predictions[app_index],axis=0).values

            plt.figure(figsize=(30,4))
            plt.plot(truth_[:1000],'r',label="Truth")
            plt.plot(pred_[:1000],'b',label="Pred")
            plt.legend()
            plt.savefig("images/%s_%s_%s_fold_%s.png"%(model_name, app_name,sequence_length,fold_number))
            plt.close()
        if save_predictions:
            truth_ = pd.concat(all_appliances_truth[app_index],axis=0).values
            pred_ = pd.concat(all_appliances_predictions[app_index],axis=0).values

            np.save("predictions/%s_fold_%s.png"%( app_name,fold_number),truth_)
            np.save("predictions/%s_%s_%s_fold_%s.png"%(model_name, app_name,sequence_length,fold_number),pred_)
        
    results = results + total_error

    results_arr.append(results)

    return all_appliances_truth, all_appliances_predictions




appliances = ["fridge",'dish washer','washing machine']
appliances.sort()

batch_size=4096
fold_numbers=[1, 2, 3]
sequence_lengths = [499, 99]
fraction_to_test = 1
cuda=True
plot=False
save_predictions=True

metrics = ['mae','f1-score','sae']
threshold = 15

"""Unpruned Model"""

create_dir_if_not_exists('results')
create_dir_if_not_exists('images')

for method in ['zero','fully_shared_mtl_iterative_pruning','fully_shared_mtl_pruning','fully_shared_mtl','unpruned_model','tensor_decomposition','normal_pruning','iterative_pruning']:

    results_arr = []
    for fold_number in fold_numbers:

        print ("Batch size:", batch_size)
        for sequence_length in sequence_lengths:

            if method=='unpruned_model':
                print ("-"*50)
                print ("Results unpruned model; sequence length: %s "%(sequence_length))
                truth, all_predictions = test_fold('unpruned_model', appliances, fold_number, sequence_length, batch_size, results_arr)
                
                print ("-"*50)
                print ("\n\n\n")
            
            elif method=='normal_pruning':
                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for %s percent Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "pruned_model_%s_percent" %(pruned_percentage)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)
                    
                    print ("-"*50)
                    print ("\n\n\n")


            elif method=='iterative_pruning':        

                for iterative_pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for %s percent Iterative Pruning; sequence length: %s "%(iterative_pruned_percentage, sequence_length))
                    model_name = "iterative_model_%s_percent" %(iterative_pruned_percentage)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)
                    
                    print ("-"*50)
                    print ("\n\n\n")

            elif method=='tensor_decomposition':

                for rank in [1,2, 4,8 ]:
                    print ("-"*50)
                    print ("Results for rank %s tensor decomposition; sequence length: %s "%(rank, sequence_length))
                    model_name = 'tensor_decomposition_rank_%s'%(rank)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                    print ("-"*50)
                    print ("\n\n\n")
            elif method == 'fully_shared_mtl':
                print ("-"*50)
                print ("Results for Fully shared MTL Model; sequence length: %s "%(sequence_length))
                model_name = method
                truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                print ("-"*50)
                print ("\n\n\n")
        
            # elif method == 'normal_mtl':
            #     print ("-"*50)
            #     print ("Results for Normal  MTL Model; sequence length: %s "%(sequence_length))
            #     model_name = method
            #     truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
            #     print ("-"*50)
            #     print ("\n\n\n")
            
            elif method=='fully_shared_mtl_pruning':

                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for Fully shared MTL %s percent Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "fully_shared_mtl_pruning_%s_percent" %(pruned_percentage)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                    
                    print ("-"*50)
                    print ("\n\n\n")

            elif method=='fully_shared_mtl_iterative_pruning':

                for pruned_percentage in [30, 60, 90]:
                    
                    print ("-"*50)
                    print ("Results for Fully shared MTL %s percent Iterative Pruning; sequence length: %s "%(pruned_percentage, sequence_length))
                    model_name = "fully_shared_mtl_iterative_model_%s_percent" %(pruned_percentage)
                    truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                    
                    print ("-"*50)
                    print ("\n\n\n")
            
            elif method=='zero':

                print ("-"*50)
                print ("Results for zero model; sequence length: %s "%( sequence_length))
                model_name = method
                truth, all_predictions = test_fold(model_name, appliances, fold_number, sequence_length, batch_size, results_arr)            
                
                print ("-"*50)
                print ("\n\n\n")



            

            
    columns  = ['Model Name',"Sequence Length","Fold Number","Batch Size"]
    for app_name in appliances:
        for metric in metrics:
            columns.append(metric+ app_name+" Error")
    
    for metric in metrics:
        columns.append("Total "+metric)
    

    results_arr= np.array(results_arr)
    df = pd.DataFrame(data=results_arr, columns=columns, index = range(len(results_arr)))
    df.to_csv(os.path.join('results','%s.csv'%(method)),index=False)
