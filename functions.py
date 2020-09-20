
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict
import time
import numpy as np
import pandas as pd
import time
import  math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import os
from sklearn.metrics import  mean_absolute_error
import math
from tensorly.decomposition import parafac
from torch.nn import Module
import json
import random
import  sys
fraction_to_train = 1

layers = ['conv1','conv2','conv3','conv4','conv5']

shuffling = True

def set_seed():
  seed = 0
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  


set_seed()

class SVDDense(Module):
  def __init__(self, A, B, bias):
    super(SVDDense, self).__init__()
    self.mat1 = torch.from_numpy(A).to(device='cuda')
    self.mat2 = torch.from_numpy(B).to(device='cuda')
    self.bias = bias

  def forward(self,x):
    x = torch.matmul(x, self.mat1)
    x = torch.matmul(x, self.mat2.T)
    x = x + self.bias
    return x



class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = self.x[index]
        y = self.y[index]

        return X, y

def create_dir_if_not_exists(dir_name_to_save_models):
  if not os.path.exists(dir_name_to_save_models):
    os.makedirs(dir_name_to_save_models)

def predict_mtl(model, app_mains, appliance_index, cuda, batch_size):


  x = app_mains
  x = torch.from_numpy(x).float()
  
  if cuda:
    x = x.to(device='cuda')
  
  predictions = []

  for index in range(math.ceil(len(x)/batch_size)):

    x_ = x[index*batch_size: (index+1)*batch_size]

    y_pred = model(x_)

    y_pred = y_pred[appliance_index].cpu().detach().numpy()

    predictions.append(y_pred)

  predictions =  np.concatenate(predictions, axis=0)

  
  
  return predictions

def compute_mtl_val_loss(model, val_mains_lst, val_appliance_lst, cuda, batch_size, parameters, appliances):
  val_pred = []
  
  for appliance_index, app_mains in enumerate(val_mains_lst):
    prediction = predict_mtl(model, app_mains, appliance_index, cuda, batch_size*10)
    val_pred.append(prediction)

  
  print ("-"*10)
  print ("Validation Loss")
  total_loss = 0  
  scaled_loss = 0
  for appliance_index, appliance_name in enumerate(appliances):
    scaled_loss+=mean_absolute_error(val_appliance_lst[appliance_index],val_pred[appliance_index])
    appliance_loss = mean_absolute_error(val_appliance_lst[appliance_index],val_pred[appliance_index]) * parameters[appliance_name]['app_std']
    total_loss+=appliance_loss
    print (appliance_name, appliance_loss)
  print ("Total Loss: %s"%(total_loss))
  print ("-"*10)
  print('\n')
  return total_loss

def train_and_save_mtl_model(models, model_name, appliances, fold_number, n_epochs, sequence_length, batch_size, opt, val_prop, all_appliances_mains_lst, all_appliances_meter_lst):
  model = models[0]
  model.cuda()

  dir_name = "fold_%s_models"%(fold_number)
  dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
  dir_name = os.path.join(dir_name, model_name)
  
  dir_name_to_save_models = dir_name
  filename_to_save_parameters = os.path.join(dir_name_to_save_models,"parameters.json")
  filename_to_save_weights = os.path.join(dir_name_to_save_models,"weights.pth")


  parameters = {}

  train_mains_lst = []
  train_appliances_lst = []

  val_mains_lst = []
  val_appliances_lst = []

  set_seed()    

  print ('-'*50)
  print ("Started Training MTL")

  for appliance_index, appliance_name in enumerate(appliances):
    set_seed()        
    train_x = all_appliances_mains_lst[appliance_index]
    train_y = all_appliances_meter_lst[appliance_index]

    n = len(train_x)

    for i in range(n):
      df = train_x[i]
      train_x[i] = df.iloc[:int(len(df)*fraction_to_train)]
  
      df = train_y[i]
      train_y[i] = df.iloc[:int(len(df)*fraction_to_train)]

    

    mains_mean  = float(np.mean(pd.concat(train_x,axis=0).values))
    mains_std = float(np.std(pd.concat(train_x,axis=0).values))

    app_mean = float(np.mean(pd.concat(train_y,axis=0).values))
    app_std = float(np.std(pd.concat(train_y,axis=0).values))

    parameters[appliance_name] = {}
    parameters[appliance_name]['mains_mean'] = mains_mean
    parameters[appliance_name]['mains_std'] = mains_std
    parameters[appliance_name]['app_mean'] = app_mean
    parameters[appliance_name]['app_std'] = app_std

    train_x = mains_preprocessing(train_x, sequence_length)
    train_y = app_preprocessing(train_y, sequence_length)

    train_x = (train_x - mains_mean)/mains_std
    train_y = (train_y - app_mean)/app_std

    indices = np.arange(len(train_x))
    if shuffling:
      np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    val_index = int(val_prop * len(train_x))

    val_x = train_x[-val_index:]
    val_y = train_y[-val_index:]

    train_x = train_x[:-val_index]
    train_y = train_y[:-val_index]

    train_mains_lst.append(train_x)
    train_appliances_lst.append(train_y)

    val_mains_lst.append(val_x)
    val_appliances_lst.append(val_y)
  
  number_of_batches_per_appliance = [len(array)//batch_size for array in train_mains_lst]
  num_batches = min(number_of_batches_per_appliance)

  num_of_minibatches_to_save_model = num_batches//4


  best_val_loss = np.inf

  criterion = nn.MSELoss()
  mae_criterion = nn.L1Loss()
  if opt=='sgd':
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
  else:
    optimizer = optim.Adam(model.parameters(), lr=0.001)

  print ("Number of training samples: %s"%(num_batches*batch_size))
  print ("Number of validation samples: %s"%(int(num_batches*batch_size*val_prop/(1-val_prop))))
  
  
  
  compute_mtl_val_loss(model, val_mains_lst, val_appliances_lst, True, batch_size, parameters, appliances)

  for n in range(n_epochs):
    epoch_loss = []
    for appliance_index, appliance_name in enumerate(appliances):
      
      arr1 = train_mains_lst[appliance_index]
      arr2 = train_appliances_lst[appliance_index]

      indices = np.arange(len(arr1))
      np.random.seed(n)
      np.random.shuffle(indices)


      arr1 = arr1[indices]
      arr2 = arr2[indices]

      
      train_mains_lst[appliance_index] = arr1
      train_appliances_lst[appliance_index] = arr2

    for batch_number in range(0, num_batches):      
      sys.stdout.flush()                
      batch_loss = []   
      loss = 0
      # zero the parameter gradients
      optimizer.zero_grad()
      for appliance_index, appliance_name in enumerate(appliances):
        inputs = train_mains_lst[appliance_index][batch_number*batch_size:(batch_number+1)*batch_size]
        labels = train_appliances_lst[appliance_index][batch_number*batch_size:(batch_number+1)*batch_size]
        
        inputs = torch.from_numpy(inputs)
        labels = torch.from_numpy(labels)
        
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')

        # forward + backward + optimize
        outputs = model(inputs)
        outputs = outputs[appliance_index]
        # print (outputs.shape, labels.shape)
        loss+=criterion(outputs, labels)
        mae_loss = mae_criterion(outputs, labels)
        batch_loss.append(round(mae_loss.item(),2))
      # print (batch_loss)      
      epoch_loss.append(batch_loss)
      # print (epoch_loss[-1])
      loss.backward()
      optimizer.step()
    
      if batch_number%num_of_minibatches_to_save_model==num_of_minibatches_to_save_model-1:              
        val_loss = compute_mtl_val_loss(model, val_mains_lst, val_appliances_lst, True, batch_size, parameters, appliances)
        if val_loss<best_val_loss:
          print ("Val Loss improved!")
          save_model(model,filename_to_save_weights)
          best_val_loss = val_loss


    mean_epoch_training_loss = np.mean(epoch_loss,axis=0)

    for appliance_index, appliance_name in enumerate(appliances):
      mean_epoch_training_loss[appliance_index] = parameters[appliance_name]['app_std'] * mean_epoch_training_loss[appliance_index]

    print ("Training Loss Epoch %s: %s"%(n+1, mean_epoch_training_loss))

    # Epoch Val loss

    val_loss = compute_mtl_val_loss(model, val_mains_lst, val_appliances_lst, True, batch_size, parameters, appliances)
    if val_loss<best_val_loss:
      print ("Val Loss improved!")
      save_model(model,filename_to_save_weights)
      best_val_loss = val_loss



    


      #         if val_loss<best_val_loss:
      #           save_model(model,filename_to_save_weights)
      #           best_val_loss = val_loss
      #           print ("Best validation loss: %s"%(best_val_loss))


  f = open(filename_to_save_parameters,'w')
  f.write(json.dumps(parameters))
  f.close()







def train_and_save_normal_model(models, model_name, appliances, fold_number, n_epochs, sequence_length, batch_size, opt, val_prop, all_appliances_mains_lst, all_appliances_meter_lst):

  dir_name = "fold_%s_models"%(fold_number)
  dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
  dir_name = os.path.join(dir_name, model_name)
  
  dir_name_to_save_models = dir_name
  filename_to_save_parameters = os.path.join(dir_name_to_save_models,"parameters.json")

  parameters = {}
  for appliance_index, appliance_name in enumerate(appliances):
    set_seed()    
    model  = models[appliance_index]
    print ('-'*50)
    print ("Started Training %s Model"%(appliance_name))
    filename_to_save_weights = os.path.join(dir_name_to_save_models,"%s.pth" %(appliance_name))
    
    train_x = all_appliances_mains_lst[appliance_index]
    train_y = all_appliances_meter_lst[appliance_index]


    n = len(train_x)

    for i in range(n):
      df = train_x[i]
      train_x[i] = df.iloc[:int(len(df)*fraction_to_train)]
  
      df = train_y[i]
      train_y[i] = df.iloc[:int(len(df)*fraction_to_train)]

    

    mains_mean  = float(np.mean(pd.concat(train_x,axis=0).values))
    mains_std = float(np.std(pd.concat(train_x,axis=0).values))

    app_mean = float(np.mean(pd.concat(train_y,axis=0).values))
    app_std = float(np.std(pd.concat(train_y,axis=0).values))

    parameters[appliance_name] = {}
    parameters[appliance_name]['mains_mean'] = mains_mean
    parameters[appliance_name]['mains_std'] = mains_std
    parameters[appliance_name]['app_mean'] = app_mean
    parameters[appliance_name]['app_std'] = app_std


    train_x = mains_preprocessing(train_x, sequence_length)
    train_y = app_preprocessing(train_y, sequence_length)

    train_x = (train_x - mains_mean)/mains_std
    train_y = (train_y - app_mean)/app_std

    indices = np.arange(len(train_x))
    if shuffling:
      np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    val_index = int(val_prop * len(train_x))

    val_x = train_x[-val_index:]
    val_y = train_y[-val_index:]

    train_x = train_x[:-val_index]
    train_y = train_y[:-val_index]


    train_x_ = torch.from_numpy(train_x).float()
    train_y_ = torch.from_numpy(train_y).float()

    trainset = Dataset(train_x_,train_y_)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=False, num_workers=1)

    best_val_loss = np.inf

    criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    if opt=='sgd':
      optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.8)
    else:
      optimizer = optim.Adam(model.parameters(), lr=0.001)

    print ("Number of training samples: %s"%(len(train_x)))
    print ("Number of validation samples: %s"%(len(val_x)))

    start_training = time.time()

    num_batches = int(len(train_x_)/batch_size)

    num_of_minibatches_to_save_model = num_batches//4

    val_pred = predict(model,val_x,True, batch_size)
    val_loss = mean_absolute_error(val_pred*app_std, val_y*app_std)
                
    print ("Epoch 0 Loss: %s"%(val_loss))
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        # if epoch==n_epochs//2:
        #   optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        #   model = torch.load(filename_to_save_weights)

        running_loss = 0.0
        
        training_losses = []
        
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data

            inputs = inputs.to('cuda')
            labels = labels.to('cuda')
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            mae_loss = mae_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            training_losses.append(mae_loss.item())

            if i%num_of_minibatches_to_save_model==num_of_minibatches_to_save_model-1:
                # print ("Checking if best")
                val_pred = predict(model,val_x,True, batch_size*10)
                val_loss = mean_absolute_error(val_pred*app_std, val_y*app_std)
                
                if val_loss<best_val_loss:
                  save_model(model,filename_to_save_weights)
                  best_val_loss = val_loss
                  print ("Best validation loss: %s"%(best_val_loss))
                  

                
        val_pred = predict(model,val_x,True, batch_size)
        val_loss = mean_absolute_error(val_pred*app_std, val_y*app_std)
        # print ("Checking if best")
        if val_loss<best_val_loss:
          save_model(model,filename_to_save_weights)
          best_val_loss = val_loss
          print ("Best validation loss: %s"%(best_val_loss))

        training_loss = np.mean(training_losses)*app_std

        print ("Epoch %s %s Training loss: %s ; Val Loss: %s"%(epoch + 1, appliance_name, training_loss, val_loss))
    end_training = time.time()
    print ("Training time:  %s seconds"%(end_training - start_training))
    print ("Finished Training %s Model"%(appliance_name))
    print ('-'*50)
    print ('\n\n\n')


  f = open(filename_to_save_parameters,'w')
  f.write(json.dumps(parameters))
  f.close()


def load_h5_file(filename, appliances):
  
  hdf = pd.HDFStore(filename,'r')
  keys = [key for key in hdf]
  keys.sort()


  all_appliances_mains_lst = []
  all_appliances_meter_lst = []


  for app_index,app_name in enumerate(appliances):
        
    app_mains_lst = []
    app_readings_lst = []
    
    app_mains_keys = [key for key in keys if (app_name in key and 'mains' in key)]
    app_data_keys = [key for key in keys if (app_name in key and 'appliance_reading' in key)]

    no_of_homes = len(app_mains_keys)

    for house in range(no_of_homes):
            
        mains_df = hdf[app_mains_keys[house]]
        app_df = hdf[app_data_keys[house]]
        
        app_mains_lst.append(mains_df)
        app_readings_lst.append(app_df)

    all_appliances_mains_lst.append(app_mains_lst)
    all_appliances_meter_lst.append(app_readings_lst)  

  return  all_appliances_mains_lst, all_appliances_meter_lst

def train_fold(models, model_name, appliances, fold_number, n_epochs, sequence_length, batch_size, opt, val_prop=0.2, num_of_minibatches_to_save_model=100, mini_batch_loss_n = 80):
  
  print ("Training for %s Epochs"%(n_epochs))

  dir_name = "fold_%s_models"%(fold_number)
  create_dir_if_not_exists(dir_name)
  
  dir_name = os.path.join(dir_name, "sequence_length_%s"%(sequence_length))
  create_dir_if_not_exists(dir_name)
  
  dir_name = os.path.join(dir_name, model_name)
  create_dir_if_not_exists(dir_name)
  
  train_file_name = 'train_%s.h5'%(fold_number)
  
  all_appliances_mains_lst, all_appliances_meter_lst = load_h5_file(train_file_name, appliances)

  training_start = time.time()
  if 'mtl' in model_name:
    train_and_save_mtl_model(models, model_name, appliances, fold_number, n_epochs, sequence_length, batch_size, opt, val_prop, all_appliances_mains_lst, all_appliances_meter_lst)
  else:
    train_and_save_normal_model(models, model_name, appliances, fold_number, n_epochs, sequence_length, batch_size, opt, val_prop, all_appliances_mains_lst, all_appliances_meter_lst)
  training_end = time.time()
  # print ("Time taken to train on fold: %s seconds"%(training_end - training_start))
  


def cp_decomposition_conv_layer(layer, rank, cuda):

    weights, [filters, channels, time] =   parafac(layer.weight.cpu().detach().numpy(), rank=rank)
    

    pointwise_s_to_r_layer = torch.nn.Conv1d(in_channels=channels.shape[0], \
            out_channels=channels.shape[1], kernel_size=1, stride=1, padding=0, 
            dilation=layer.dilation, bias=False)

    depthwise_horizontal_layer = torch.nn.Conv1d(in_channels=time.shape[1], 
            out_channels=time.shape[1], 
            kernel_size=time.shape[0], stride=layer.stride,
            padding=0, dilation=layer.dilation, groups=channels.shape[1], bias=False)

    pointwise_r_to_t_layer = torch.nn.Conv1d(in_channels=filters.shape[1], 
            out_channels=filters.shape[0], kernel_size=1, stride=1,
            padding=0, dilation=layer.dilation, bias=True)

    pointwise_r_to_t_layer.bias.data = layer.bias.data


    
    channels = torch.from_numpy(channels)
    time = torch.from_numpy(time)
    filters = torch.from_numpy(filters)

    pointwise_s_to_r_layer.weight.data = \
        torch.transpose(channels, 1, 0).unsqueeze(-1)

    depthwise_horizontal_layer.weight.data = \
        torch.transpose(time, 1, 0).unsqueeze(1)

    pointwise_r_to_t_layer.weight.data = filters.unsqueeze(-1)

    new_layers = [pointwise_s_to_r_layer,depthwise_horizontal_layer, pointwise_r_to_t_layer]
    
    model = nn.Sequential(*new_layers)
    

    if cuda:
      model.cuda()
    return model


def set_new_dense_weights(layer, new_weights, new_biases):
  layer.weight = nn.Parameter(new_weights)
  layer.bias = nn.Parameter(new_biases)
  
  # layer.weight = nn.Parameter(torch.ones_like(new_weights))
  # layer.bias = nn.Parameter(torch.ones_like(new_biases))

  # for i in range(new_weights.shape[0]):
  #   for j in range(new_weights.shape[1]):
  #       layer.weight[i,j] = new_weights[i,j]
  
  # for i in range(new_biases.shape[0]):
  #   layer.bias[i] = new_biases[i]

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x, y):
        'Initialization'
        self.x = x
        self.y = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        X = self.x[index]
        y = self.y[index]

        return X, y

def save_model(model,filename):
  torch.save(model, filename)




def set_new_conv_weights(layer, new_weights, new_biases):
  layer.weight = nn.Parameter(new_weights)
  layer.bias = nn.Parameter(new_biases)

  # layer.weight = nn.Parameter(torch.ones_like(new_weights))
  # layer.bias = nn.Parameter(torch.ones_like(new_biases))

  # for i in range(new_weights.shape[0]):
  #   for j in range(new_weights.shape[1]):
  #     for k in range(new_weights.shape[2]):
  #       layer.weight[i,j,k] = new_weights[i,j,k]
  
  # for i in range(new_biases.shape[0]):
  #   layer.bias[i] = new_biases[i]


def remove_lowest_neurons(model, layer_name, num_neurons_to_remove):
  
  layers =  ['conv1','conv2','conv3','conv4','conv5','fc1','fc2']
  
  
  current_layer_index = layers.index(layer_name)
  layer = getattr(model, layer_name)

  original_shape = layer.weight.shape

  current_layer_weights = layer.weight
  current_layer_biases = layer.bias

  if hasattr(model, 'app_layers'):

    next_layers = getattr(model, 'app_layers')
    next_layer_weights = [next_model.fc2.weight for next_model in next_layers]
    next_layer_biases = [next_model.fc2.bias for next_model in next_layers]
    num_appliances = len(model.app_layers)

  else:

    next_layer = getattr(model, layers[current_layer_index+1])
    next_layer_weights = next_layer.weight
    next_layer_biases = next_layer.bias


  
  for n in range(num_neurons_to_remove):
    l1_neurons_norm = torch.sum(torch.abs(current_layer_weights),axis=1)
    min_neuron_index = torch.argmin(l1_neurons_norm).item()
    #Removing the neuron in the current layer
    new_neuron_weights = torch.cat([current_layer_weights[0:min_neuron_index], current_layer_weights[min_neuron_index+1:]])
    new_neuron_biases = torch.cat([current_layer_biases[0:min_neuron_index], current_layer_biases[min_neuron_index+1:]])

    if hasattr(model, 'app_layers'):
      new_next_layer_weights = []

      for appliance_index in range(num_appliances):
        appliance_weights = next_layer_weights[appliance_index]
        new_appliance_weights = torch.cat([appliance_weights[:,0:min_neuron_index], appliance_weights[:,min_neuron_index+1:]],axis=1)
        
        new_next_layer_weights.append(new_appliance_weights)
      

    else:
      new_next_layer_weights = torch.cat([next_layer_weights[:,0:min_neuron_index], next_layer_weights[:,min_neuron_index+1:]],axis=1)
    
    
    current_layer_weights = new_neuron_weights
    current_layer_biases = new_neuron_biases
    next_layer_weights = new_next_layer_weights
    next_layer_biases = next_layer_biases

  if layer_name =='fc1':
    with torch.no_grad():
      if hasattr(model, 'app_layers'):
        for appliance_index in range(num_appliances):
          next_layer = model.app_layers[appliance_index].fc2
          set_new_dense_weights(layer, current_layer_weights, current_layer_biases)
          set_new_dense_weights(next_layer, new_next_layer_weights[appliance_index], next_layer_biases[appliance_index])


      else:
        set_new_dense_weights(layer, current_layer_weights, current_layer_biases)
        set_new_dense_weights(next_layer, new_next_layer_weights, next_layer_biases)




def remove_lowest_filters(model, layer_name, num_filters_to_remove):
  
  layers =  ['conv1','conv2','conv3','conv4','conv5','fc1','fc2']
  
  
  current_layer_index = layers.index(layer_name)
  layer = getattr(model, layer_name)

  original_shape = layer.weight.shape


  current_layer_weights = layer.weight
  current_layer_biases = layer.bias


  next_layer = getattr(model, layers[current_layer_index+1])
  next_layer_weights = next_layer.weight
  next_layer_biases = next_layer.bias



  for n in range(num_filters_to_remove):
    l1_filters_norm = torch.sum(torch.sum(torch.abs(current_layer_weights),axis=2),axis=1)
    min_filter_index = torch.argmin(l1_filters_norm).item()
    #Removing the filter in the current layer
    new_filter_weights = torch.cat([current_layer_weights[0:min_filter_index], current_layer_weights[min_filter_index+1:]])
    new_filter_biases = torch.cat([current_layer_biases[0:min_filter_index], current_layer_biases[min_filter_index+1:]])

    #Removing the weights in the next layer

    if layer_name!="conv5":
        # print (current_layer_weights.shape, new_filter_weights.shape)
        new_next_layer_weights = torch.cat([next_layer_weights[:,0:min_filter_index], next_layer_weights[:,min_filter_index+1:]],axis=1)

    else:
        output_n_channels = new_filter_weights.shape[0] + 1
        output_sequence_length = next_layer_weights.shape[1]//output_n_channels
        new_next_layer_weights = torch.cat([next_layer_weights[:,0:min_filter_index*output_sequence_length], next_layer_weights[:,output_sequence_length*(min_filter_index+1):]],axis=1)

    current_layer_weights = new_filter_weights
    current_layer_biases = new_filter_biases
    next_layer_weights = new_next_layer_weights
    next_layer_biases = next_layer_biases
  with torch.no_grad():
    set_new_conv_weights(layer, new_filter_weights, new_filter_biases)

  if layer_name!="conv5":
      with torch.no_grad():
          set_new_conv_weights(next_layer, new_next_layer_weights, next_layer_biases)
  else:
      with torch.no_grad():
        set_new_dense_weights(next_layer, new_next_layer_weights, next_layer_biases)

def remove_filters_and_neurons(model, percent_to_remove):
  
  for i in ['conv1','conv2','conv3','conv4','conv5']:
    original_shape = getattr(model, i).weight.shape
    num_filters_to_remove = math.ceil(percent_to_remove * original_shape[0])
    remove_lowest_filters(model, i, num_filters_to_remove)
  
  original_shape = model.fc1.weight.shape
  number_of_neurons_to_remove = math.ceil(percent_to_remove * original_shape[0])
  remove_lowest_neurons(model, 'fc1', number_of_neurons_to_remove)

def remove_filters_only(model, percent_to_remove):
    for i in ['conv1','conv2','conv3','conv4','conv5']:
      original_shape = getattr(model, i).weight.shape
      num_filters_to_remove = math.ceil(percent_to_remove * original_shape[0])
      remove_lowest_filters(model, i, num_filters_to_remove)

def remove_neurons_only(model, percent_to_remove):
    original_shape = model.fc1.weight.shape
    number_of_neurons_to_remove = math.ceil(percent_to_remove * original_shape[0])
    remove_lowest_neurons(model, 'fc1', number_of_neurons_to_remove)
      

def iteratively_remove(model, num_filters_to_remove, number_of_neurons_to_remove):
  for i, filter_name in enumerate(['conv1','conv2','conv3','conv4','conv5']):
    remove_lowest_filters(model, filter_name, num_filters_to_remove[i])
  remove_lowest_neurons(model, 'fc1', number_of_neurons_to_remove)

def global_pruning(model, percent_to_remove):
  total_num_filters = 0

  for layer in layers:
    total_num_filters+=getattr(model, layer).weight.data.shape[0]
  
  num_filters_to_remove = int(percent_to_remove * total_num_filters)
  
  


  filter_norms = []
  for layer in layers:
    current_layer_weights = getattr(model, layer).weight.data
    filters_l1_norm = torch.sum(torch.sum(torch.abs(current_layer_weights),axis=2),axis=1)    

    for filter_norm in filters_l1_norm:
      filter_norms.append([filter_norm.item(), layer])
  
  filter_norms.sort()
  

  print (filter_norms[num_filters_to_remove])

  for n in range(num_filters_to_remove):
    current_filter_norm = filter_norms[n][0]
    current_layer = filter_norms[n][1]
    remove_lowest_filters(model, current_layer, 1)

  for layer in layers:
    current_layer_weights = getattr(model, layer).weight.data
    print (current_layer_weights.shape)
  


def cp_decompose_model(model, rank):

  

  new_layers = []

  for layer in layers:
      new_layers.append(cp_decomposition_conv_layer(getattr(model,layer),rank,True))

  model.conv1 = new_layers[0]
  model.conv2 = new_layers[1]
  model.conv3 = new_layers[2]
  model.conv4 = new_layers[3]
  model.conv5 = new_layers[4]

  
  # Doing dense compression  

  weights, [A,B] = parafac(model.fc1.weight.cpu().detach().numpy().T,rank=rank)
  bias = model.fc1.bias

  model.fc1 = SVDDense(A, B, bias)
  model.fc1.cuda()



def mains_preprocessing(mains_lst,sequence_length):
    mains_df_list = []
    for mains in mains_lst:
        
        new_mains = mains.values.flatten()
        n = sequence_length
        units_to_pad = n // 2
        new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
        new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
        mains_df_list.append(new_mains)
    
    mains = np.concatenate(mains_df_list,axis=0)
    mains = mains.reshape((-1,1,sequence_length))
    return mains

def app_preprocessing(app_lst,sequence_length):
    return pd.concat(app_lst,axis=0).values.reshape((-1,1))



def predict(model, x, cuda, batch_size):

  x = torch.from_numpy(x).float()
  
  if cuda:
    x = x.to(device='cuda')
  
  predictions = []

  for index in range(math.ceil(len(x)/batch_size)):

    x_ = x[index*batch_size: (index+1)*batch_size]

    y_pred = model(x_)

    y_pred = y_pred.cpu().detach().numpy()

    predictions.append(y_pred)

  return np.concatenate(predictions, axis=0)


def sparsity(model):
  
  attributes = ['conv1','conv2','conv3','conv4','conv5','fc1','fc2']

  sparse_params = 0
  total_params = 0

  for attr in attributes:

    layer = getattr(model,attr)

    sparse_entries  = float(torch.sum(layer.weight == 0))
    n_elements = float(layer.weight.nelement())
    
    sparse_params+=sparse_entries
    total_params+=n_elements

    print ("Sparsity in %s layer is %0.3f "%(attr, 100*sparse_entries/n_elements))
    
  print ("Total sparsity is %s "%(100 * sparse_params/total_params))
  
  
