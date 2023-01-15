import torch
from graphrl.models.metapath2vec import Metapath2vec
import torch.optim as optim
from arguments import parse_args

from graphrl.utils.log_utils import *
from graphrl.dataloader.graph.heterogeneous import GraphLoader, generate_metapath
from graphrl.utils.train_utils import *
from tqdm import tqdm


def train(args):
    fix_seed(args.seed)
    args.dataset = 'net_dbis'
    args.num_walks_per_node = 8
    args.walk_length = 10
    args.min_count = 5
    args.care_type = 0
    args.window_size = 5
    args.batch_size = 512
    args.lr = 0.01
    data_path = f"{args.data_dir}/{args.dataset}"

    generate_metapath(data_path, args.num_walks_per_node, args.walk_length)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(
        log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load data
    data = GraphLoader(data_path, args.min_count, args.care_type, args.window_size)
    

    # construct model & optimizer
    model = Metapath2vec(data)

    print("Device {}".format(device))
    model.to(device)

    logging.info(model)
    torch.autograd.set_detect_anomaly(True)

    optimizer = optim.SparseAdam(model.parameters(), lr=args.lr)
    

    model(n_epochs=args.n_epoch, batch_size=args.batch_size ,optimizer=optimizer, logging=logging)



def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
