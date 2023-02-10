import torch
from graphrl.models.gnn import GraphSAGE
from arguments import parse_args
import torch.optim as optim
import torch.nn.functional as F
import os

from graphrl.utils.log_utils import *
from graphrl.dataloader.graph.homogeneous import GraphLoader
from graphrl.utils.train_utils import *
from tqdm import tqdm


def train(model, optimizer, features, adj_matrix, labels, epochs, idx_train, idx_val):
    max_val_acc = 0
    
    for epoch in tqdm(range(1, epochs + 1)):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj_matrix)
        loss_train = F.nll_loss(output[idx_train], labels[idx_train])
        acc_train = calculate_accuracy(output[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = calculate_accuracy(output[idx_val], labels[idx_val])
        print('Epoch: {:04d}'.format(epoch),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'loss_val: {:.4f}'.format(loss_val.item()),
            'acc_val: {:.4f}'.format(acc_val.item()))
        if max_val_acc < acc_val.item():
            max_val_acc = acc_val.item()

    print(f"Max accuracy: {max_val_acc}")

def test(model, features, adj_matrix, labels):
    model.eval()
    output = model(features, adj_matrix)
    accuracy = calculate_accuracy(output, labels)
    print("Test set results:",
        "accuracy= {:.4f}".format(accuracy.item()))


def main():
    args = parse_args()
    fix_seed(args.seed)

    save_dir = 'result/{}'.format(os.path.basename(__file__))
    args.save_dir = save_dir

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(
        log_save_id), no_console=False)
    logging.info(args)

    args.dataset = 'cora_v2'
    args.features_file = 'cora.content'
    args.edges_file = 'cora.cites'
    args.dropout = 0.0
    args.num_layers = 3
    args.hidden_dim = 32
    args.weight_decay = 1e-3

    # GPU / CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load data
    data = GraphLoader(data_dir= args.data_dir, dataset=args.dataset, directed=False)
    data.load_features(args.features_file)
    data.load_adj_matrix_from_edges(args.edges_file)

    idx_train = range(1000)
    idx_val = range(1000, 1200)
    idx_test = range(1200, 1500)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    features = data.features
    adj_matrix = data.adj_matrix
    labels = data.labels

    num_classes = labels.max().item() + 1


    # construct model & optimizer
    model = GraphSAGE(num_features=features.shape[1], hidden_dim=args.hidden_dim, num_classes=num_classes, num_layers=args.num_layers, dropout=args.dropout)

    optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

    print("Device {}".format(device))
    if device == torch.device("cuda"):
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)
        features = features.to(device)
        adj_matrix = adj_matrix.to(device)
        labels = labels.to(device)
        model.to(device)

    logging.info(model)
    train(model, optimizer, features, adj_matrix, labels, args.n_epoch, idx_train, idx_val)


if __name__ == '__main__':
    main()
