from connector.models.base import BaseModel
from connector.models.skipgram import SkipGram
from tqdm import tqdm
import torch

class Metapath2vec(BaseModel):
    def __init__(self, data, dim=128, device="cpu", logging=None):
        super(Metapath2vec, self).__init__()
        self.data = data
        self.embedding_size = len(data.word2id)
        self.embedding_dim = dim
        self.device = device
        self.logging = logging
        self.skip_gram = SkipGram(self.embedding_size , dim)

    def forward(self, n_epochs, batch_size, optimizer, logging=None):
        running_loss= 0.0
        loss_list = []
        data_batches = self.data.sample_batch_data(batch_size)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(data_batches))
        
        for iter in tqdm(range(1, n_epochs + 1)):
            running_loss= 0.0

            for index, sample_batched in enumerate(data_batches):
                if len(sample_batched[0]) < 1:
                    continue

                scheduler.step()
                optimizer.zero_grad()
                u_pos = sample_batched[0].to(self.device)
                v_pos = sample_batched[1].to(self.device)
                v_neg = sample_batched[2].to(self.device)
                loss = self.skip_gram(u_pos, v_pos, v_neg)

                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            running_loss = running_loss / len(data_batches)
            loss_list.append(running_loss)

            if logging is not None:
                logging.info(f"Epoch  -------")
                # Logging every epoch
                logging.info("Metapath2vec model loss {}".format(running_loss))
                logging.info("Loss list {}".format(loss_list))


    def save_embedding(self, data, output_file):
        self.skip_gram.save_embedding(data.id2word, output_file)
            
