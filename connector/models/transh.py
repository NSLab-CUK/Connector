import torch
import torch.nn as nn
import torch.nn.functional as F
from connector.utils.model_utils import _L2_loss_mean
from connector.models.base import BaseModel


class TransH(BaseModel):

    def __init__(self, ent_dim, rel_dim, n_entities, n_relations, l2loss_lambda=1e-5, device="cpu"):
        
        super(TransH, self).__init__()
        self.device = device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = ent_dim
        self.relation_dim = rel_dim

        self.l2loss_lambda = l2loss_lambda

        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        self.normalize_vector = nn.Embedding(self.n_relations, self.relation_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)
        nn.init.xavier_uniform_(self.normalize_vector.weight)

    def transfer(self, x, norm):
        norm = F.normalize(norm, p = 2, dim = -1)
        if x.shape[0] != norm.shape[0]:
            x = x.view(-1, norm.shape[0], x.shape[-1])
            norm = norm.view(-1, norm.shape[0], norm.shape[-1])
            x = x - torch.sum(x * norm, -1, True) * norm
            return x.view(-1, x.shape[-1])
        else:
            return x - torch.sum(x * norm, -1, True) * norm

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)

        all_embed = self.entity_embed.weight

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)
        r_norm = self.normalize_vector(r)

        h_transfer = self.transfer(head_embed, r_norm)
        t_pos_transfer = self.transfer(tail_pos_embed, r_norm)
        t_neg_transfer = self.transfer(tail_neg_embed, r_norm)

        # Trans R
        pos_score = torch.sum(
            torch.pow(h_transfer + r_embed - t_pos_transfer, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(h_transfer + r_embed - t_neg_transfer, 2), dim=1)  # (kg_batch_size)

        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(h_transfer) + _L2_loss_mean(r_embed) + _L2_loss_mean(t_pos_transfer) + _L2_loss_mean(
            t_neg_transfer)
        loss = triplet_loss + self.l2loss_lambda * l2_loss
        return loss

    def forward(self, *input):
        return self.calc_triplet_loss(*input)