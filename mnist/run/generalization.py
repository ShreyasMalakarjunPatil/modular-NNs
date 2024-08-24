import os
import torch
import pickle as pkl

from utils import train, load
from models import utils


def run(args):

    load.set_seed(args.seed)
    dev = load.load_device(args.gpu)
    direct_models, direct_results = load.create_directory(args)

    if args.model == 'mlp' or args.model == 'cnn':
        path = direct_results + args.model + str(args.arch) + str(args.pruning_ratio) + str(args.lr) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.num_digits) + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.num_samples_per_combination_train) + str(args.num_samples_per_combination_test) + str(args.normalize_img) + '.pkl'
    elif args.model == 'hierarchically_modular_mlp' or args.model == 'hierarchically_modular_cnn':
        path = direct_results + args.model + str(args.num_img_modules) + str(args.num_modules) + str(args.img_module_arch) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.lr) + str(args.lr_module_input) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.num_digits) + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.num_samples_per_combination_train) + str(args.num_samples_per_combination_test) + str(args.normalize_img) + '.pkl'
    elif args.model == 'hierarchically_modular_shared_modules_mlp' or args.model == 'hierarchically_modular_shared_modules_cnn':
        path = direct_results + args.model + str(args.num_img_slots) + str(args.num_slots) + str(args.num_img_shared_modules) + str(args.num_shared_modules) + str(args.img_module_arch) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.known_module_locations) + str(args.lr) + str(args.lr_module_input) + str(args.lr_module_location) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.num_digits) + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.num_samples_per_combination_train) + str(args.num_samples_per_combination_test) + str(args.normalize_img) + '.pkl'

    if os.path.exists(path):
        print('Shreyas:', path)
    else:
        train_loader, val_loader, test_loader1, test_loader2 = load.load_data(args.dataset_path, args.modularity, args.batch_size, args.num_digits, args.num_samples_per_combination_train, args.num_samples_per_combination_test, args.dataset_split, args.dataset_split_seed, args.normalize_img, args.workers)
        loss = load.load_loss(args.loss)
        opt, opt_kwargs = load.load_optimizer(args.optimizer)

        if args.model == 'mlp' or args.model == 'cnn':

            model = load.load_model(args.model)(args.arch).to(dev)
            if args.pruning_ratio > 0.0:
                weight_mask, bias_mask = utils.rand_prune_masks(model, args.pruning_ratio, dev)
                model.set_masks(weight_mask, bias_mask)
            optimizer = opt(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
        
        elif args.model == 'hierarchically_modular_mlp' or args.model == 'hierarchically_modular_cnn':

            if args.known_module_inputs:

                module_inp_indices = load.load_routing(args)
                model = load.load_model(args.model)(args.num_img_modules, args.num_modules, args.img_module_arch, args.module_arch, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau, args.known_module_inputs, module_inp_indices).to(dev)
                optimizer = opt([
                    {'params': model.img_modules.parameters(), 'lr': args.lr},
                    {'params': model.module_mlps.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)

            else:

                model = load.load_model(args.model)(args.num_img_modules, args.num_modules, args.img_module_arch, args.module_arch, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau).to(dev)
                optimizer = opt([
                    {'params': model.module_inp_embeddings, 'lr': args.lr_module_input},
                    {'params': model.img_modules.parameters(), 'lr': args.lr},
                    {'params': model.module_mlps.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)

        elif args.model == 'hierarchically_modular_shared_modules_mlp' or args.model == 'hierarchically_modular_shared_modules_cnn':

            if args.known_module_inputs and args.known_module_locations:

                module_input_indices, slot_module_indices = load.load_routing(args)
                model = load.load_model(args.model)(args.num_img_shared_modules, args.num_shared_modules, args.num_img_slots, args.num_slots, args.img_module_arch, args.module_arch, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau, args.known_module_inputs, args.known_module_locations, module_input_indices, slot_module_indices).to(dev)
                optimizer = opt([
                    {'params': model.module_mlps.parameters(), 'lr': args.lr},
                    {'params': model.img_modules.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)
        
            else:    
            
                model = load.load_model(args.model)(args.num_img_shared_modules, args.num_shared_modules, args.num_img_slots, args.num_slots, args.img_module_arch, args.module_arch, args.modularity[0], args.modularity[-1], args.topk, args.use_gumbel, args.per_sample_routing, args.tau).to(dev)
                optimizer = opt([
                    {'params': model.module_inp_embeddings, 'lr': args.lr_module_input},
                    {'params': model.module_loc_embeddings, 'lr': args.lr_module_location},
                    {'params': model.module_mlps.parameters(), 'lr': args.lr},
                    {'params': model.img_modules.parameters(), 'lr': args.lr}
                    ], weight_decay=args.weight_decay, **opt_kwargs)

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.gamma, last_epoch=- 1)
        (model, validation_loss, test_loss1, test_loss2, validation_accuracy, test_accuracy1, test_accuracy2) = train.train_network(model, loss, optimizer, train_loader, val_loader, test_loader1, test_loader2, dev, args.epochs, scheduler)

        results = []
        results.append(validation_loss)
        results.append(test_loss1)
        results.append(test_loss2)
        results.append(validation_accuracy)
        results.append(test_accuracy1)
        results.append(test_accuracy2)

        if args.model == 'mlp' or args.model == 'cnn':

            with open(direct_results + args.model + str(args.arch) + str(args.pruning_ratio) + str(args.lr) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.num_digits) + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.num_samples_per_combination_train) + str(args.num_samples_per_combination_test) + str(args.normalize_img) + '.pkl', "wb") as fout:
                pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)


        if args.model == 'hierarchically_modular_mlp' or args.model == 'hierarchically_modular_cnn':

            with open(direct_results + args.model + str(args.num_img_modules) + str(args.num_modules) + str(args.img_module_arch) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.lr) + str(args.lr_module_input) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.num_digits) + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.num_samples_per_combination_train) + str(args.num_samples_per_combination_test) + str(args.normalize_img) + '.pkl', "wb") as fout:
                pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)

        if args.model == 'hierarchically_modular_shared_modules_mlp' or args.model == 'hierarchically_modular_shared_modules_cnn':

            with open(direct_results + args.model + str(args.num_img_slots) + str(args.num_slots) + str(args.num_img_shared_modules) + str(args.num_shared_modules) + str(args.img_module_arch) + str(args.module_arch) + str(args.topk) + str(args.use_gumbel) + str(args.tau) + str(args.per_sample_routing) + str(args.known_module_inputs) + str(args.known_module_locations) + str(args.lr) + str(args.lr_module_input) + str(args.lr_module_location) + str(args.batch_size) + str(args.epochs) + str(args.weight_decay) + str(args.gamma) + str(args.seed) + '_Results' + str(args.num_digits) + str(args.dataset_split) + str(args.dataset_split_seed) + str(args.num_samples_per_combination_train) + str(args.num_samples_per_combination_test) + str(args.normalize_img) + '.pkl', "wb") as fout:
                pkl.dump(results, fout, protocol=pkl.HIGHEST_PROTOCOL)