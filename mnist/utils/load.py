import torch
import numpy as np
import os
import pickle as pkl
import copy
import random

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.utils.data.sampler import SubsetRandomSampler

import torch.optim as optim

from models.mlp import mlp  
from models.hierarchically_modular_mlp import hierarchically_modular_mlp  
from models.hierarchically_modular_shared_modules_mlp import hierarchically_modular_shared_modules_mlp
from models.cnn import cnn  
from models.hierarchically_modular_cnn import hierarchically_modular_cnn
from models.hierarchically_modular_shared_modules_cnn import hierarchically_modular_shared_modules_cnn

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_device(gpu):
    use_cuda = torch.cuda.is_available()
    print('Use Cuda',use_cuda)
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

def generate_pairs(num_range, dataset_split, dataset_split_seed):

    all_pairs = [(i, j) for i in num_range for j in num_range]

    if dataset_split<1.0:

        rng = random.Random(dataset_split_seed)
        rng.shuffle(all_pairs)
    
        num_pairs_to_select = int(len(all_pairs) * dataset_split)

        train_pairs = all_pairs[:num_pairs_to_select]
        test_pairs1 = all_pairs[:num_pairs_to_select]
        test_pairs2 = all_pairs[num_pairs_to_select:]
        
        l = len(test_pairs2)
        val_pairs = test_pairs2[:l//2]
        test_pairs2 = test_pairs2[l//2:]

    else:

        train_pairs, test_pairs1, test_pairs2, val_pairs = all_pairs, all_pairs, all_pairs, all_pairs
    
    return train_pairs, val_pairs, test_pairs1, test_pairs2

class LimitedPairMNIST(Dataset):
    def __init__(self, mnist_dataset, label_mapping, valid_label_pairs, num_samples_per_combination=1000, dataset_split_seed=40):

        self.mnist_dataset = mnist_dataset
        self.label_mapping = label_mapping
        self.valid_label_pairs = valid_label_pairs
        self.num_samples_per_combination = num_samples_per_combination
        self.dataset_split_seed = dataset_split_seed

        self.label_to_indices = self._create_label_to_indices()
        self.pairs = self._generate_filtered_pairs()

    def _create_label_to_indices(self):

        label_to_indices = defaultdict(list)
        targets_tensor = torch.tensor(self.mnist_dataset.targets.clone().detach())
        
        for idx, label in enumerate(targets_tensor):
            label_to_indices[label.item()].append(idx)
        
        return label_to_indices

    def _generate_filtered_pairs(self):

        rng = random.Random(self.dataset_split_seed)

        pairs = []

        for label1, label2 in self.valid_label_pairs:

            if label1 in self.label_to_indices and label2 in self.label_to_indices:

                indices1 = self.label_to_indices[label1]
                indices2 = self.label_to_indices[label2]

                possible_pairs = [(i, j) for i in indices1 for j in indices2]
                selected_pairs = rng.sample(possible_pairs, min(self.num_samples_per_combination, len(possible_pairs)))

                pairs.extend(selected_pairs)
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2 = self.pairs[idx]
        img1, label1 = self.mnist_dataset[idx1]
        img2, label2 = self.mnist_dataset[idx2]
        return torch.cat((img1, img2)), self.label_mapping[label1][label2]

def load_data(dataset_path, modularity, batch_size, num_digits = 8, num_samples_per_combination_train = 1000, num_samples_per_combination_test = 100, dataset_split=None, dataset_split_seed=None, normalize_img=False, num_workers=4):

    num_range = range(num_digits)
    train_pairs, val_pairs, test_pairs1, test_pairs2 = generate_pairs(num_range, dataset_split, dataset_split_seed)
    print('Shreyas Pair Sizes: ', len(train_pairs), len(val_pairs), len(test_pairs1), len(test_pairs2))
    print(train_pairs)
    print(val_pairs)
    print(test_pairs1)
    print(test_pairs2)

    path = dataset_path + str(modularity) + '/mnist' + str(modularity) + '.pkl'
    with open(path, "rb") as fout:
        lable_mapping = pkl.load(fout)

    if normalize_img==True:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_dataset_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_dataset = LimitedPairMNIST(mnist_dataset, lable_mapping, train_pairs, num_samples_per_combination=num_samples_per_combination_train, dataset_split_seed=dataset_split_seed)
    val_dataset = LimitedPairMNIST(mnist_dataset, lable_mapping, val_pairs, num_samples_per_combination=num_samples_per_combination_test, dataset_split_seed=dataset_split_seed)
    
    test_dataset1 = LimitedPairMNIST(mnist_dataset_test, lable_mapping, test_pairs1, num_samples_per_combination=num_samples_per_combination_test, dataset_split_seed=dataset_split_seed)
    test_dataset2 = LimitedPairMNIST(mnist_dataset_test, lable_mapping, test_pairs2, num_samples_per_combination=num_samples_per_combination_test, dataset_split_seed=dataset_split_seed)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle=True,
                                                    num_workers = num_workers)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers)
    test_loader1 = torch.utils.data.DataLoader(dataset=test_dataset1,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers)
    test_loader2 = torch.utils.data.DataLoader(dataset=test_dataset2,
                                                    batch_size=batch_size,
                                                    shuffle=False,
                                                    num_workers=num_workers)
    
    return train_loader, val_loader, test_loader1, test_loader2

def load_routing(args):

    if (args.model == 'hierarchically_modular_cnn' and args.known_module_inputs) or (args.model == 'hierarchically_modular_mlp' and args.known_module_inputs):

        dataset_name = args.dataset_path + str(args.modularity) + '/mnist' + str(args.modularity) + '_module_input_indices.pkl'
        with open(dataset_name, "rb") as fout:
            module_input_indices = pkl.load(fout)
            
        return module_input_indices
    
    if (args.model == 'hierarchically_modular_shared_modules_cnn' and args.known_module_inputs and args.known_module_locations) or (args.model == 'hierarchically_modular_shared_modules_mlp'  and args.known_module_inputs and args.known_module_locations):

        dataset_name = args.dataset_path + str(args.modularity) + '/mnist' + str(args.modularity) + '_module_input_indices_shared_modules.pkl'
        with open(dataset_name, "rb") as fout:
            module_input_indices = pkl.load(fout)

        dataset_name = args.dataset_path + str(args.modularity) + '/mnist' + str(args.modularity) + '_slot_module_indices_shared_modules.pkl'
        with open(dataset_name, "rb") as fout:
            slot_module_indices = pkl.load(fout)
        
        return module_input_indices, slot_module_indices

def load_model(model):
    models = {
            'mlp' : mlp,
            'hierarchically_modular_mlp' : hierarchically_modular_mlp,
            'hierarchically_modular_shared_modules_mlp' : hierarchically_modular_shared_modules_mlp,
            'cnn' : cnn,
            'hierarchically_modular_cnn' : hierarchically_modular_cnn,
            'hierarchically_modular_shared_modules_cnn' : hierarchically_modular_shared_modules_cnn
            }

    return models[model]

def load_optimizer(optimizer):
    optimizers = {
        'adam' : (optim.Adam, {}),
        'sgd' : (optim.SGD, {}),
        'momentum' : (optim.SGD, {'momentum' : 0.9, 'nesterov' : True}),
        'rms' : (optim.RMSprop, {})
    }
    return optimizers[optimizer]

def load_loss(loss_function):
    losses = {
            'BCE' : torch.nn.BCELoss(),
            'MSE' : torch.nn.MSELoss(),
            'CE' : torch.nn.CrossEntropyLoss()
        }
    return losses[loss_function]

def create_directory(args):

    direct_models = args.result_path + args.experiment + '/'  + str(args.modularity) + '/Models/' +str(args.model) + '/'
    isdir = os.path.isdir(direct_models)

    if not isdir:
        os.makedirs(direct_models)

    direct_results = args.result_path + args.experiment + '/'  + str(args.modularity) + '/Results/' +str(args.model)+'/'
    isdir = os.path.isdir(direct_results)

    if not isdir:
        os.makedirs(direct_results)
        
    return direct_models, direct_results