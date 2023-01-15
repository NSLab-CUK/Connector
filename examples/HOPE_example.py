import torch
from graphrl.models.hope import HOPE
from arguments import parse_args
import os

from graphrl.utils.log_utils import *
from graphrl.dataloader.graph.homogeneous import GraphLoader
from graphrl.utils.train_utils import *


def train(args):
    fix_seed(args.seed)

    save_dir = 'result/{}'.format(os.path.basename(__file__))
    args.save_dir = save_dir

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(
        log_save_id), no_console=False)
    logging.info(args)

    args.beta = 0.01
    args.alpha = 1e-2
    args.measurement = 'katz'
    args.dataset = 'cora'

    # GPU / CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load data
    data = GraphLoader(data_dir= args.data_dir, dataset=args.dataset, directed=True)
    data.read_edgelist()

    # construct model & optimizer
    model = HOPE(args)

    print("Device {}".format(device))
    model.to(device)
    logging.info(model)
    
    embedding = model(data, args.measurement)

    print(embedding)




def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
