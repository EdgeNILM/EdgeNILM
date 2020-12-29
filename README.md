
# EdgeNILM: Towards NILM on Edge Devices

This is the codebase our paper published in [Buildsys 2020].


We have used the `trainer.py` script to train the model, you can use it in the following way
  -    `python trainer.py unpruned_model`
  -    `python trainer.py normal_pruning`
  -    `python trainer.py iterative_pruning`
  -    `python trainer.py tensor_decomposition`
  -    `python trainer.py fully_shared_mtl`
  -    `python trainer.py fully_shared_mtl_pruning`
  -    `python trainer.py fully_shared_mtl_iterative_pruning`
  
**Please note that, in order to execute the `normal_pruning`,`iterative_pruning` and `tensor_decomposition`, you first need to train the `unpruned_model`**

**Please note that, in order to execute the `fully_shared_mtl_pruning` and `fully_shared_mtl_iterative_pruning`, you first need to train the `fully_shared_mtl`**


You can find the models we used to train here (Link will be posted to the google drive by November 30, 2020).

**Once you finish training the model or after downloading the pre-trained models, you can do the following:**

By executing `python test.py` we get the results of all the models for each fold for all the appliances.

You can execute `python flops_compute.py` to create the CSV file which contains the  info about the number of MFLOPs needed for a single pass.

You can execute `python time_compute.py` to create the CSV file which contains the  info about the inference time and model size. 

  
