import torch.optim as optim
from connector.dataloader.graph.signed import SignedGraph
from connector.models.sine import SiNE
from arguments import parse_args
from tqdm import tqdm
from connector.utils.log_utils import *

def train_model(model, data, args, logging=None):
    triplets = data.get_triplets()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    min_loss = 1000

    for iter in tqdm(range(args.n_epoch)):
        model.zero_grad()
        n_i, n_j, n_k = data.sample_batch(triplets, args.training_batch_size)

        loss = model(n_i, n_j, n_k, args.delta)

        regularizer_loss = args.alpha * model.regularize_weights()

        loss += regularizer_loss

        loss.backward()
        optimizer.step()

        loss_value = float(loss)

        if min_loss > loss_value:
            min_loss = loss_value

        if logging:
            logging.info(f"Epoch {iter} - Loss: {loss_value}")

    if logging:
        logging.info(f"Min loss = {min_loss}")

    return model


def main():
    args = parse_args()
    args.dim1= 128
    args.dim2= 64
    args.bias1 = 0
    args.bias2 = 0
    args.delta = 1
    args.alpha = 1
    args.lr = 0.1
    args.n_epoch = 1000
    args.weight_decay = 0.0
    args.dataset = 'epinion'

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(
        log_save_id), no_console=False)

    data_file = f"{args.data_dir}/{args.dataset}/data.txt"

    data = SignedGraph.load_from_file(data_file)

    num_nodes = data.num_nodes

    model = SiNE(num_nodes, dim1=args.dim1, dim2=args.dim2)

    train_model(model, data, args, logging)



if __name__ == '__main__':
    main()