# When and how are modular networks better?

This repository is the official implementation of the paper titled "When and how are modular networks better?" under review at ICLR 2025. 

Visualizations of architectures we explored in our paper

![Alt text](images/models.png)




To install requirements run:

```setup
pip3 install -r requirements.txt
```

## Training commands

The code is divided into 2 directories, one for learning Boolean functions and the second to learn the MNIST based hierarchical and modular task. To run the training scripts please move into the respective directories. 

### Training on Boolean functions

* An example command to run MLP training: 

```
python3 main.py --experiment generalization --num_tasks 1 --dataset_path ./datasets/generalization/ --result_path ./results/ --modularity 6 3 2 2 --model mlp --arch 6 36 36 36 2 --pruning_ratio 0.0 --lr 0.1 --batch_size 64 --epochs 1000 --seed 1 
```

* Running training for hierarchically modular NNs 
```
python3 main.py --experiment generalization --num_tasks 1 --dataset_path ./datasets/generalization/ --result_path ./results/ --modularity 6 3 2 2 --model hierarchically_modular --num_modules 3 2 --module_arch 2 12 1 --lr 0.01 --lr_module_input 0.1 --batch_size 8 --epochs 200 --seed 1 
```

* To run training for known routing simply add the following arguments: 
```
 --known_module_input
```
* To use Gumbel noise based structural sampling: 
```
 --use_gumbel --tau 1.0 --per_sample_routing
```
* Running training for hierarchically modular NNs with shared modules
```
python3 main.py --experiment generalization --num_tasks 1 --dataset_path ./datasets/generalization/ --result_path ./results/ --modularity 6 3 2 2 --model hierarchically_modular_shared_modules --num_shared_modules 2 --num_slots 3 2 --module_arch 2 12 1 --lr 0.01 --lr_module_input 0.1 --lr_module_location 0.1 --batch_size 8 --epochs 200 --seed 1 
```
* To run training for known routing simply add the following arguments: 
```
 --known_module_input --known_module_locations
```


### Training on Boolean functions

* An example command to run MLP training: 

```
python3 main.py --experiment generalization --num_tasks 1 --dataset_path ./datasets/generalization/ --result_path ./results/ --modularity 6 3 2 2 --model mlp --arch 6 36 36 36 2 --pruning_ratio 0.0 --lr 0.1 --batch_size 64 --epochs 1000 --seed 1 
```

* Running training for hierarchically modular NNs 
```
python3 main.py --experiment generalization --num_tasks 1 --dataset_path ./datasets/generalization/ --result_path ./results/ --modularity 6 3 2 2 --model hierarchically_modular --num_modules 3 2 --module_arch 2 12 1 --lr 0.01 --lr_module_input 0.1 --batch_size 8 --epochs 200 --seed 1 
```

* To run training for known routing simply add the following arguments: 
```
 --known_module_input
```
* To use Gumbel noise based structural sampling: 
```
 --use_gumbel --tau 1.0 --per_sample_routing
```
* Running training for hierarchically modular NNs with shared modules
```
python3 main.py --experiment generalization --num_tasks 1 --dataset_path ./datasets/generalization/ --result_path ./results/ --modularity 6 3 2 2 --model hierarchically_modular_shared_modules --num_shared_modules 2 --num_slots 3 2 --module_arch 2 12 1 --lr 0.01 --lr_module_input 0.1 --lr_module_location 0.1 --batch_size 8 --epochs 200 --seed 1 
```
* To run training for known routing simply add the following arguments: 
```
 --known_module_input --known_module_locations
```



## Code details

### run

* The file generalization.py loads the data, model, runs the training and saves the results.


### models


### utils

The directory utils consists of 5 files:

* load.py -> this script is used across experiments and runs to load models, datasets, optimizers, loss functions etc. on the basis of the arguments input to the parser
* train.py -> runs the training iterations across all experiments

## Other arguments and hyper-parameters

Find a full list of hyper-parameters and values through:
```
python3 main.py --help
```
