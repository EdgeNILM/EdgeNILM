
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import random
import os
import numpy as np

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

hidden_layer_dropout = 0.2



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
  


def dummy_network(sequence_length,cuda):
    # Model architecture
    set_seed()
    
    model = torch.nn.Sequential(
        
    torch.nn.Conv1d(out_channels = 30 , kernel_size=10, in_channels = 1),
    torch.nn.ReLU(),

    torch.nn.Conv1d(out_channels = 30 , kernel_size=8, in_channels = 30),
    torch.nn.ReLU(),

    torch.nn.Conv1d(out_channels = 40 , kernel_size=6, in_channels = 30),
    torch.nn.ReLU(),

    torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 40),
    torch.nn.ReLU(), 
    torch.nn.Dropout(p=hidden_layer_dropout),

    torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 50),
    torch.nn.ReLU(), 
    torch.nn.Dropout(p=hidden_layer_dropout),

    torch.nn.Flatten(),
    )
    if cuda:
      model.cuda()
    return model




class Seq2Point(nn.Module):
  def __init__(self, sequence_length, cuda):
    set_seed()
    super(Seq2Point, self).__init__()
    dummy_model = dummy_network(sequence_length,cuda)
    rand_tensor = torch.randn(1, 1, sequence_length)
    
    if cuda:
      rand_tensor = rand_tensor.to(device='cuda')
    dummy_output = dummy_model(rand_tensor)
    num_of_flattened_neurons = dummy_output.shape[-1]

    ## Now define the actual network

    self.conv1 = torch.nn.Conv1d(out_channels = 30 , kernel_size=10, in_channels = 1)
    self.conv2 = torch.nn.Conv1d(out_channels = 30 , kernel_size=8, in_channels = 30)
    self.conv3 = torch.nn.Conv1d(out_channels = 40 , kernel_size=6, in_channels = 30)
    self.conv4 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 40)
    self.conv5 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 50)
    self.fc1 = torch.nn.Linear(out_features=1024, in_features=num_of_flattened_neurons)
    self.fc2 = torch.nn.Linear(out_features=1, in_features=1024)
    self.dropout1 = torch.nn.Dropout(hidden_layer_dropout)
    self.dropout2 = torch.nn.Dropout(hidden_layer_dropout)
    if cuda:
      self.cuda()

  def forward(self, X):
    x = self.conv1(X)
    x = F.relu(x)

    x = self.conv2(x)
    x = F.relu(x)

    x = self.conv3(x)
    x = F.relu(x)

    x = self.conv4(x)
    x = F.relu(x)

    x = self.dropout1(x)

    x = self.conv5(x)
    x = F.relu(x)
    x = self.dropout2(x)

    x = x.reshape(x.size(0), -1)
    
    x = self.fc1(x)
    x = F.relu(x)

    x = self.fc2(x)
    return x

# MTL with common convolution layers

class mtl_single(nn.Module):
  def __init__(self, sequence_length, cuda):
    set_seed()
    super(mtl_single, self).__init__()

    dummy_model = dummy_network(sequence_length,cuda)
    rand_tensor = torch.randn(1, 1, sequence_length)
    
    if cuda:
      rand_tensor = rand_tensor.to(device='cuda')
    dummy_output = dummy_model(rand_tensor)
    num_of_flattened_neurons = dummy_output.shape[-1]
    self.conv5 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 50)
    self.drop1 = torch.nn.Dropout(0.2)
    self.fc1 = torch.nn.Linear(out_features=1024, in_features=num_of_flattened_neurons)
    
    self.fc2 = torch.nn.Linear(out_features=1, in_features=1024)
    

    if cuda:
      self.cuda()
  
  def forward(self, x):
    x = self.conv5(x)
    x = F.relu(x)
    x = self.drop1(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc1(x)
    x = F.relu(x)
    x = self.fc2(x)
    return x


class NotFullySharedMTL(nn.Module):
  def __init__(self, sequence_length, num_app, cuda):
    set_seed()
    super(NotFullySharedMTL, self).__init__()


    self.conv1 = torch.nn.Conv1d(out_channels = 30 , kernel_size=10, in_channels = 1)
    self.conv2 = torch.nn.Conv1d(out_channels = 30 , kernel_size=8, in_channels = 30)
    self.conv3 = torch.nn.Conv1d(out_channels = 40 , kernel_size=6, in_channels = 30)
    self.conv4 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 40)
    self.drop = torch.nn.Dropout(0.2)
    self.app_layers = nn.ModuleList([mtl_single(sequence_length, cuda) for j in range(num_app)])

    self.num_app = num_app

    if cuda:
      self.cuda()
  
  def forward(self, x):
    x = self.conv1(x)
    x = F.relu(x)

    x = self.conv2(x)
    x = F.relu(x)

    x = self.conv3(x)
    x = F.relu(x)

    x = self.conv4(x)
    x = F.relu(x)

    x = self.drop(x)
    
    outs = [self.app_layers[j](x) for j in range(self.num_app)]

    return outs


# Fully shared model

class FinalDense(nn.Module):
  def __init__(self, sequence_length, cuda):
    set_seed()
    super(FinalDense, self).__init__()
    self.fc2 = torch.nn.Linear(out_features=1, in_features=1024)
  
  def forward(self, x):
    return self.fc2(x)


class FullySharedMTL(nn.Module):
  def __init__(self, sequence_length, num_app, cuda):
    set_seed()
    super(FullySharedMTL, self).__init__()

    self.num_app = num_app

    dummy_model = dummy_network(sequence_length,cuda)
    rand_tensor = torch.randn(1, 1, sequence_length)
    
    if cuda:
      rand_tensor = rand_tensor.to(device='cuda')
    dummy_output = dummy_model(rand_tensor)
    num_of_flattened_neurons = dummy_output.shape[-1]

    ## Now define the actual network

    self.conv1 = torch.nn.Conv1d(out_channels = 30 , kernel_size=10, in_channels = 1)
    self.conv2 = torch.nn.Conv1d(out_channels = 30 , kernel_size=8, in_channels = 30)
    self.conv3 = torch.nn.Conv1d(out_channels = 40 , kernel_size=6, in_channels = 30)
    self.conv4 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 40)
    self.conv5 = torch.nn.Conv1d(out_channels = 50 , kernel_size=5, in_channels = 50)
    self.fc1 = torch.nn.Linear(out_features=1024, in_features=num_of_flattened_neurons)
    self.app_layers = nn.ModuleList([FinalDense(sequence_length, cuda) for j in range(num_app)])
    self.dropout1 = torch.nn.Dropout(hidden_layer_dropout)
    self.dropout2 = torch.nn.Dropout(hidden_layer_dropout)
    if cuda:
      self.cuda()

  def forward(self, X):
    x = self.conv1(X)
    x = F.relu(x)

    x = self.conv2(x)
    x = F.relu(x)

    x = self.conv3(x)
    x = F.relu(x)

    x = self.conv4(x)
    x = F.relu(x)

    x = self.dropout1(x)

    x = self.conv5(x)
    x = F.relu(x)
    x = self.dropout2(x)

    x = x.reshape(x.size(0), -1)
    
    x = self.fc1(x)
    x = F.relu(x)
    outputs = []
    for i in range(self.num_app):
      y = self.app_layers[i](x)
      outputs.append(y)
    return outputs


