
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Run Framework.")

    parser.add_argument('--exp_name', type=str, default="run")
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed.')

    parser.add_argument('--dataset', nargs='?', default='FB13',
                        help='Choose a dataset')
    parser.add_argument('--data_dir', nargs='?', default='../datasets/',
                        help='Input data path.')

    parser.add_argument('--training_batch_size', type=int, default=1000,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='Test batch size (the head number to test every batch).')

    parser.add_argument('--entity_dim', type=int, default=300,
                        help='head / tail Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=300,
                        help='Relation Embedding size.')

    parser.add_argument('--dim', type=int, default=300,
                        help='Node Embedding size.')

    


    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=20,
                        help='Number of epoch.')
    parser.add_argument('--epoch_data_rate', type=float, default=1,
                        help='Sampling data rate for each epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--kg_print_every', type=int, default=500,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--training_neg_rate', type=int, default=1,
                        help='The training negative rate.')

    parser.add_argument('--device', nargs='?', default='cpu',
                        help='Choose a device to run')
                        
    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')

    args = parser.parse_args()

    args.dataset = args.dataset.replace("'", "")

    save_dir = 'result/{}'.format(os.path.basename(__file__))
    args.save_dir = save_dir

    return args

