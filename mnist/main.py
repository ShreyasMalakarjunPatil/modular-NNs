import argparse

from run import generalization

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical and modular neural networks')
    parser.add_argument('--experiment', type=str, default='generalization',
                        choices=['generalization'])

    ####################################################################################################################
    # Specify Task, Function, Path to dataset
    ####################################################################################################################

    parser.add_argument('--modularity', nargs='+', type=int, default= [2,2,3,2,2])
    parser.add_argument('--dataset_path', type=str, default='./datasets/generalization/')

    parser.add_argument('--num_digits', type=int, default=8)
    parser.add_argument('--dataset_split', type=float, default=1.0)
    parser.add_argument('--dataset_split_seed', type=int, default=123)

    parser.add_argument('--num_samples_per_combination_train', type=int, default=100)
    parser.add_argument('--num_samples_per_combination_test', type=int, default=10000)
    parser.add_argument('--normalize_img', type=bool, default=False)

    parser.add_argument('--result_path', type=str, default='./results/')

    ####################################################################################################################
    # Specify NN architecture and training hyper-parameters
    ####################################################################################################################

    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'cnn', 'hierarchically_modular_mlp', 'hierarchically_modular_shared_modules_mlp', 'hierarchically_modular_cnn', 'hierarchically_modular_shared_modules_cnn'])

    ### MLP and CNN architecture arguments
    parser.add_argument('--arch', nargs='+', type=int, default=[784*2,256,128,36,36,36,36,2])
    parser.add_argument('--pruning_ratio', type=float, default=0.0)

    ### Hierarchically modular architecture arguments
    parser.add_argument('--num_img_modules', nargs='+', type=int, default=[2])
    parser.add_argument('--num_modules', nargs='+', type=int, default=[3,2])

    ### Hierarchically modular architecture with shared modules arguments
    parser.add_argument('--num_img_slots', nargs='+', type=int, default=[2])
    parser.add_argument('--num_slots', nargs='+', type=int, default=[3,2])
    parser.add_argument('--num_img_shared_modules', type=int, default=1)
    parser.add_argument('--num_shared_modules', type=int, default=2)

    ### Shared arguments
    parser.add_argument('--img_module_arch', nargs='+', type=int, default=[784,128,64,3])
    parser.add_argument('--module_arch', nargs='+', type=int, default=[2,12,1])

    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--use_gumbel', type=bool, default=False)
    parser.add_argument('--per_sample_routing', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=None)

    parser.add_argument('--known_module_inputs', type=bool, default=False)
    parser.add_argument('--known_module_locations', type=bool, default=False)

    ####################################################################################################################
    # Specify training hyper-parameters
    ####################################################################################################################

    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'momentum', 'rms'])
    parser.add_argument('--loss', type=str, default='BCE',
                        choices=['CE', 'MSE', 'BCE'])

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_module_input', type=float, default=None)
    parser.add_argument('--lr_module_location', type=float, default=None)

    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--scheduler', type=str, default = 'exp', 
                        choices = ['exp', 'cosine'])
    parser.add_argument('--gamma', type=float, default = 1.0)

    ####################################################################################################################
    # Specify runtime hyper-parameters
    ####################################################################################################################

    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    print(args)

    if args.experiment == 'generalization':
        generalization.run(args)