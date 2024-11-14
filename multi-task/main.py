import argparse

from run import multi_task

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hierarchical and modular neural networks')
    parser.add_argument('--experiment', type=str, default='multi_task',
                        choices=['multi_task'])

    ####################################################################################################################
    # Specify Task, Function, Path to dataset
    ####################################################################################################################

    parser.add_argument('--num_tasks', type=int, default=2)
    parser.add_argument('--modularity', nargs='+', type=int, default= [6,3,2,2])
    parser.add_argument('--dataset_path', type=str, default='./datasets/generalization/')
    parser.add_argument('--dataset_noise', type=bool, default=True)
    parser.add_argument('--dataset_split', type=float, default=0.5)
    parser.add_argument('--dataset_split_seed', type=int, default=40)
    
    parser.add_argument('--result_path', type=str, default='./results2/')

    ####################################################################################################################
    # Specify ANN architecture and training hyper-parameters
    ####################################################################################################################

    parser.add_argument('--model', type=str, default='hierarchically_modular_shared_modules',
                        choices=['mlp','mlp2','hierarchically_modular', 'hierarchically_modular_shared_modules'])

    ### MLP architecture arguments
    parser.add_argument('--arch', nargs='+', type=int, default=[6,36,36,36,2])
    parser.add_argument('--pruning_ratio', type=float, default=0.0)

    ### Hierarchically modular architecture arguments
    parser.add_argument('--num_modules', nargs='+', type=int, default=[6,4])

    ### Hierarchically modular architecture with shared modules arguments
    parser.add_argument('--num_slots', nargs='+', type=int, default=[3,2])
    parser.add_argument('--num_shared_modules', type=int, default=3)
    
    ### Shared arguments
    parser.add_argument('--module_arch', nargs='+', type=int, default=[2,12,1])
    parser.add_argument('--topk', type=int, default=2)
    parser.add_argument('--use_gumbel', type=bool, default=False)
    parser.add_argument('--per_sample_routing', type=bool, default=False)
    parser.add_argument('--tau', type=float, default=None)

    parser.add_argument('--known_func_outputs', type=bool, default=False)
    parser.add_argument('--known_module_inputs', type=bool, default=False)
    parser.add_argument('--known_module_locations', type=bool, default=False)


    ####################################################################################################################
    # Specify training hyper-parameters
    ####################################################################################################################

    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['sgd', 'adam', 'momentum', 'rms'])
    parser.add_argument('--loss', type=str, default='BCE',
                        choices=['CE', 'MSE', 'BCE'])

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=1000)

    parser.add_argument('--lr', type=float, default=0.01)
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

    if args.experiment == 'multi_task':
        multi_task.run(args)