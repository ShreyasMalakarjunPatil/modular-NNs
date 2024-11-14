import torch
import numpy as np
import os
import pickle as pkl
import copy

from torch.utils.data import Dataset, ConcatDataset
import torch.optim as optim

from models.mlp import mlp, mlp2  
from models.hierarchically_modular_nn import hierarchically_modular  
from models.hierarchically_modular_shared_modules import hierarchically_modular_shared_modules

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def load_device(gpu):
    use_cuda = torch.cuda.is_available()
    print('Use Cuda',use_cuda)
    return torch.device(("cuda:" + str(gpu)) if use_cuda else "cpu")

def AddGaussianNoise(tensor, std):
    return tensor + torch.randn(tensor.size()) * std

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, task_id = None, transform=None, std=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
        self.task_id = task_id
        self.std = std
    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x, self.std)
        y = self.tensors[1][index]
        return x, y, self.task_id
    def __len__(self):
        return self.tensors[0].size(0)

def load_data(dataset_path, num_tasks, modularity, batch_size, dataset_noise=False, dataset_split=None, dataset_split_seed=None):

    path = dataset_path + str(modularity) + '/boolean' + str(modularity) + '_'

    train_dataset_set = []
    train_dataset2_set = []
    val_dataset_set = []
    test_dataset_set = []

    generator = torch.Generator().manual_seed(dataset_split_seed)

    for i in range(num_tasks):
        dataset_path = path + str(i) + '.pkl'
        with open(dataset_path, "rb") as fout:
            data = pkl.load(fout)
        X = data[0]
        Y = data[1]

        if dataset_noise:
            dataset = CustomTensorDataset(tensors=(X, Y), task_id=i, transform=AddGaussianNoise, std=0.1)
        else:
            dataset = CustomTensorDataset(tensors=(X, Y), task_id=i)

        train_size = int(data[0].size()[0]*dataset_split)
        test_size = int(data[0].size()[0]) - int(data[0].size()[0]*dataset_split)
        val_size = test_size//2
        test_size = test_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=generator)
        train_dataset2 = copy.deepcopy(train_dataset)

        if dataset_noise:            
            test_dataset = copy.deepcopy(test_dataset)
            val_dataset = copy.deepcopy(val_dataset)
            val_dataset.dataset.transform = None
            test_dataset.dataset.transform = None
            train_dataset2.dataset.transform = None

        train_dataset_set.append(train_dataset)
        train_dataset2_set.append(train_dataset2)
        test_dataset_set.append(test_dataset)
        val_dataset_set.append(val_dataset)

        print('Shreyas dataset sizes : ',len(train_dataset), len(test_dataset), len(val_dataset))

    train_dataset = ConcatDataset(train_dataset_set)
    train_dataset2 = ConcatDataset(train_dataset2_set)
    val_dataset = ConcatDataset(val_dataset_set)
    test_dataset = ConcatDataset(test_dataset_set)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = 2)
    train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,
                                                    batch_size=128,
                                                    shuffle=False,
                                                    num_workers=2)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                    batch_size=128,
                                                    shuffle=False,
                                                    num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                    batch_size=128,
                                                    shuffle=False,
                                                    num_workers=2)

    return train_loader, train_loader2, val_loader, test_loader

def load_routing(args):

    if (args.model == 'hierarchically_modular' and args.known_module_inputs) or (args.model == 'hierarchically_modular' and args.known_func_outputs):

        module_input_indices = []
        for i in range(args.num_tasks):
            dataset_name = args.dataset_path + str(args.modularity) + '/boolean' + str(args.modularity) + '_' + str(i) + '_module_input_indices.pkl'
            with open(dataset_name, "rb") as fout:
                a = pkl.load(fout)
            module_input_indices.append(a)
            
        return module_input_indices
    
    if (args.model == 'hierarchically_modular_shared_modules' and args.known_module_inputs and args.known_module_locations) or (args.model == 'hierarchically_modular_shared_modules' and args.known_func_outputs):

        module_input_indices = []
        slot_module_indices = []

        for i in range(args.num_tasks):

            dataset_name = args.dataset_path + str(args.modularity) + '/boolean' + str(args.modularity) + '_' + str(i) + '_module_input_indices_shared_modules.pkl'
            with open(dataset_name, "rb") as fout:
                a = pkl.load(fout)
            module_input_indices.append(a)

            dataset_name = args.dataset_path + str(args.modularity) + '/boolean' + str(args.modularity) + '_' + str(i) + '_slot_module_indices_shared_modules.pkl'
            with open(dataset_name, "rb") as fout:
                a = pkl.load(fout)
            slot_module_indices.append(a)
            
        return module_input_indices, slot_module_indices

def load_model(model):
    models = {
            'mlp' : mlp,
            'mlp2' : mlp2,
            'hierarchically_modular' : hierarchically_modular,
            'hierarchically_modular_shared_modules' : hierarchically_modular_shared_modules
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