import torch
import numpy as np
import random
from graphrl.models.transr import TransR
import torch.optim as optim
from tqdm import tqdm
from time import time
import sys
from arguments import parse_args

from graphrl.utils.log_utils import *
from graphrl.dataloader.graph.knowledge import GraphLoader
from graphrl.utils.train_utils import *


def train(args):
    fix_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(
        log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # load data
    data = GraphLoader(dataset=args.dataset, data_dir=args.data_dir, training_batch_size=args.training_batch_size, device=device, logging=logging)

    # construct model & optimizer
    model = TransR(ent_dim=args.entity_dim, rel_dim=args.relation_dim, n_entities=data.n_entities, n_relations=data.n_relations, device=device)

    print("Device {}".format(device))
    model.to(device)

    logging.info(model)
    torch.autograd.set_detect_anomaly(True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train
    loss_kg_list = []

    kg_time_training = []

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))


    # Train model
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        time3 = time()
        kg_total_loss = 0

        # Sampling data for each epoch
        n_data_samples = int(len(list(data.train_kg_dict)) * args.epoch_data_rate)
        epoch_sampling_data_list = random.sample(list(data.train_kg_dict), n_data_samples)
        epoch_sampling_data_dict = {k: data.train_kg_dict[k] for k in epoch_sampling_data_list}
        n_kg_batch = n_data_samples // data.training_batch_size + 1

        for iter in tqdm(range(1, n_kg_batch + 1), desc=f"EP:{epoch}_train"):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(
                epoch_sampling_data_dict, data.training_batch_size, data.n_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            optimizer.zero_grad()
            kg_batch_loss = model(kg_batch_head, kg_batch_relation,
                                  kg_batch_pos_tail, kg_batch_neg_tail)

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
                                                                                                  n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            optimizer.step()
            kg_total_loss += kg_batch_loss.item()

            if iter % 50 == 0:
                torch.cuda.empty_cache()

            loss_value = kg_total_loss / n_kg_batch

            if (iter % args.kg_print_every) == 0:
                logging.info(
                    'Training: Epoch {:04d}/{:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, args.n_epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(),
                                                               kg_total_loss / iter))
        logging.info(
            'Pre-training: Epoch {:04d}/{:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                epoch, args.n_epoch, n_kg_batch, time() - time3, loss_value))

        loss_kg_list.append(loss_value)
        kg_time_training.append(time() - time3)

        torch.cuda.empty_cache()

        # Logging every epoch
        logging.info("Loss TransR {}".format(loss_kg_list))
        logging.info("Training TransR time {}".format(kg_time_training))
    
    logging.info("FINALLL -------")
    # Logging every epoch
    logging.info("TransR loss list {}".format(loss_kg_list))
    logging.info("TransR time training {}".format(kg_time_training))



def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
