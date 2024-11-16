import torch
import pickle as pkl

from utils import train, load
from models import utils
import os
import copy

def run(args):

    load.set_seed(args.seed)
    dev = load.load_device(args.gpu)
    direct_models, direct_results = load.create_directory(args)

    global_arch = copy.deepcopy(args.arch)

    if args.model=='mlp' or args.model=='mlp2':
        path = direct_results + args.model + str(args.arch) + str(args.pruning_ratio) + str(args.lr) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.dataset_split) + str(args.dataset_split_seed)+ str(args.dataset_noise) + '.pkl'
    if args.model=='hierarchically_modular' or args.model=='hierarchically_modular2':
        path = direct_results + args.model + str(args.num_modules) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.known_func_outputs) + str(args.lr) + str(args.lr_module_input) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.dataset_noise) + '.pkl'
    if args.model=='hierarchically_modular_shared_modules' or args.model=='hierarchically_modular_shared_modules2':
        path = direct_results + args.model + str(args.num_slots) + str(args.num_shared_modules) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.known_module_locations) + str(args.known_func_outputs) + str(args.lr) + str(args.lr_module_input) + str(args.lr_module_location) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.dataset_noise) + '.pkl'
    
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print('Shreyas:', path)
    else:
        train_loader, train_loader2, validation_loader, test_loader = load.load_data(args.dataset_path, args.num_tasks, args.modularity, args.batch_size, args.dataset_noise, args.dataset_split, args.dataset_split_seed)
        loss = load.load_loss(args.loss)
        opt, opt_kwargs = load.load_optimizer(args.optimizer)

        if args.model == 'mlp' or args.model=='mlp2':

            model = load.load_model(args.model)(args.arch, args.num_tasks).to(dev)
            if args.pruning_ratio > 0.0:
                weight_mask, bias_mask = utils.rand_prune_masks(model, args.pruning_ratio, dev)
                model.set_masks(weight_mask, bias_mask)
            optimizer = opt(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)

        elif args.model == 'hierarchically_modular' or args.model=='hierarchically_modular2':

            if args.known_module_inputs:

                module_inp_indices = load.load_routing(args)
                model = load.load_model(args.model)(args.num_modules, args.module_arch, args.num_tasks, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau, args.known_module_inputs, module_inp_indices).to(dev)
                optimizer = opt([
                    {'params': model.module_mlps.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)

            else:

                model = load.load_model(args.model)(args.num_modules, args.module_arch, args.num_tasks, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau).to(dev)
                optimizer = opt([
                    {'params': model.module_inp_embeddings, 'lr': args.lr_module_input},
                    {'params': model.module_mlps.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)

        elif args.model == 'hierarchically_modular_shared_modules' or args.model=='hierarchically_modular_shared_modules2':

            if args.known_module_inputs and args.known_module_locations:

                module_input_indices, slot_module_indices = load.load_routing(args)
                model = load.load_model(args.model)(args.num_shared_modules, args.num_slots, args.module_arch, args.num_tasks, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau, args.known_module_inputs, args.known_module_locations, module_input_indices, slot_module_indices).to(dev)
                optimizer = opt([
                    {'params': model.module_mlps.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)

            elif args.known_func_outputs:

                module_input_indices, slot_module_indices = load.load_routing(args)
                model = load.load_model(args.model)(args.num_shared_modules, args.num_slots, args.module_arch, args.num_tasks, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau, args.known_module_inputs, args.known_module_locations, module_input_indices, slot_module_indices, args.known_func_outputs).to(dev)
                optimizer = opt([
                    {'params': model.module_inp_embeddings[:-1], 'lr': args.lr_module_input},
                    {'params': model.module_loc_embeddings, 'lr': args.lr_module_location},
                    {'params': model.module_mlps.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)
        
            else:    
            
                model = load.load_model(args.model)(args.num_shared_modules, args.num_slots, args.module_arch, args.num_tasks, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau).to(dev)
                optimizer = opt([
                    {'params': model.module_inp_embeddings, 'lr': args.lr_module_input},
                    {'params': model.module_loc_embeddings, 'lr': args.lr_module_location},
                    {'params': model.module_mlps.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=- 1)
        (model, validation_loss, train_loss, test_loss, validation_accuracy, train_accuracy, test_accuracy) = train.train_network(model, loss, optimizer, train_loader, train_loader2, validation_loader, test_loader, dev, args.epochs, scheduler)

        results = []
        results.append(train_loss)
        results.append(validation_loss)
        results.append(test_loss)
        results.append(train_accuracy)
        results.append(validation_accuracy)
        results.append(test_accuracy)
        

        if args.model=='mlp' or args.model=='mlp2':

            with open(direct_results + args.model + str(global_arch) + str(args.pruning_ratio) + str(args.lr) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.dataset_split) + str(args.dataset_split_seed)+ str(args.dataset_noise) + '.pkl', "wb") as fout:
                pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)


        if args.model=='hierarchically_modular' or args.model=='hierarchically_modular2':

            with open(direct_results + args.model + str(args.num_modules) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.known_func_outputs) + str(args.lr) + str(args.lr_module_input) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.dataset_noise) + '.pkl', "wb") as fout:
                pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)

        if args.model=='hierarchically_modular_shared_modules' or args.model=='hierarchically_modular_shared_modules2':

            with open(direct_results + args.model + str(args.num_slots) + str(args.num_shared_modules) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.known_module_locations) + str(args.known_func_outputs) + str(args.lr) + str(args.lr_module_input) + str(args.lr_module_location) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.dataset_noise) + '.pkl', "wb") as fout:
                pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)