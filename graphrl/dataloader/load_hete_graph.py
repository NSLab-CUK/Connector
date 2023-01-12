import numpy as np
import torch
import tqdm
import dgl
import os

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class GraphLoader(object):
    NEGATIVE_TABLE_SIZE = 1e8

    def __init__(self, data_dir, min_count, care_type, window_size=8):

        self.negatives = []
        self.discards = []
        self.negpos = 0
        self.care_type = care_type
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.inputFileName = os.path.join(data_dir, 'data.txt')
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()
        self.dataset = ModelDataset(self, window_size)

    def read_words(self, min_count):
        word_frequency = dict()
        for line in open(self.inputFileName, encoding="ISO-8859-1"):
            line = line.split()
            if len(line) > 1:
                self.sentences_count += 1
                for word in line:
                    if len(word) > 0:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

                        if self.token_count % 1000000 == 0:
                            print(
                                "Read " + str(int(self.token_count / 1000000)) + "M words.")

        wid = 0
        for w, c in word_frequency.items():
            if c < min_count:
                continue
            self.word2id[w] = wid
            self.id2word[wid] = w
            self.word_frequency[wid] = c
            wid += 1

        self.word_count = len(self.word2id)
        print("Total embeddings: " + str(len(self.word2id)))

    def initTableDiscards(self):
        # get a frequency table for sub-sampling. Note that the frequency is adjusted by
        # sub-sampling tricks.
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)

    def initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * GraphLoader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count):
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def sample_batch_data(self, batch_size, shuffle=True, number_workers=0):
        return DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle, num_workers=number_workers, collate_fn=self.dataset.collate)

    def getNegatives(self, target, size):  # TODO check equality with target
        if self.care_type == 0:
            response = self.negatives[self.negpos:self.negpos + size]
            self.negpos = (self.negpos + size) % len(self.negatives)
            if len(response) != size:
                return np.concatenate((response, self.negatives[0:self.negpos]))
        return response


class ModelDataset(Dataset):
    def __init__(self, data, window_size):
        # read in data, window_size and input filename
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf-8")

    def __len__(self):
        # return the number of walks
        return self.data.sentences_count

    def __getitem__(self, idx):
        # return the list of pairs (center, context, 5 negatives)
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                words = line.split()

                if len(words) > 1:
                    word_ids = [self.data.word2id[w] for w in words if
                                w in self.data.word2id and np.random.rand() < self.data.discards[self.data.word2id[w]]]

                    pair_catch = []
                    for i, u in enumerate(word_ids):
                        for j, v in enumerate(
                                word_ids[max(i - self.window_size, 0):i + self.window_size]):
                            assert u < self.data.word_count
                            assert v < self.data.word_count
                            if i == j:
                                continue
                            pair_catch.append(
                                (u, v, self.data.getNegatives(v, 5)))
                    return pair_catch

    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _ in batch if len(batch) > 0]
        all_v = [v for batch in batches for _, v, _ in batch if len(batch) > 0]
        all_neg_v = [neg_v for batch in batches for _,
                     _, neg_v in batch if len(batch) > 0]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)
        

def construct_graph(path):
    paper_ids = []
    paper_names = []
    author_ids = []
    author_names = []
    conf_ids = []
    conf_names = []
    f_3 = open(os.path.join(path, "id_author.txt"), encoding="ISO-8859-1")
    f_4 = open(os.path.join(path, "id_conf.txt"), encoding="ISO-8859-1")
    f_5 = open(os.path.join(path, "paper.txt"), encoding="ISO-8859-1")
    while True:
        z = f_3.readline()
        if not z:
            break
        z = z.strip().split()
        identity = int(z[0])
        author_ids.append(identity)
        author_names.append(z[1])
    while True:
        w = f_4.readline()
        if not w:
            break
        w = w.strip().split()
        identity = int(w[0])
        conf_ids.append(identity)
        conf_names.append(w[1])
    while True:
        v = f_5.readline()
        if not v:
            break
        v = v.strip().split()
        identity = int(v[0])
        paper_name = 'p' + ''.join(v[1:])
        paper_ids.append(identity)
        paper_names.append(paper_name)
    f_3.close()
    f_4.close()
    f_5.close()

    author_ids_invmap = {x: i for i, x in enumerate(author_ids)}
    conf_ids_invmap = {x: i for i, x in enumerate(conf_ids)}
    paper_ids_invmap = {x: i for i, x in enumerate(paper_ids)}

    paper_author_src = []
    paper_author_dst = []
    paper_conf_src = []
    paper_conf_dst = []
    f_1 = open(os.path.join(path, "paper_author.txt"), "r")
    f_2 = open(os.path.join(path, "paper_conf.txt"), "r")
    for x in f_1:
        x = x.split('\t')
        x[0] = int(x[0])
        x[1] = int(x[1].strip('\n'))
        paper_author_src.append(paper_ids_invmap[x[0]])
        paper_author_dst.append(author_ids_invmap[x[1]])
    for y in f_2:
        y = y.split('\t')
        y[0] = int(y[0])
        y[1] = int(y[1].strip('\n'))
        paper_conf_src.append(paper_ids_invmap[y[0]])
        paper_conf_dst.append(conf_ids_invmap[y[1]])
    f_1.close()
    f_2.close()

    hg = dgl.heterograph({
        ('paper', 'pa', 'author'): (paper_author_src, paper_author_dst),
        ('author', 'ap', 'paper'): (paper_author_dst, paper_author_src),
        ('paper', 'pc', 'conf'): (paper_conf_src, paper_conf_dst),
        ('conf', 'cp', 'paper'): (paper_conf_dst, paper_conf_src)})
    return hg, author_names, conf_names, paper_names


def generate_metapath(data_path, num_walks_per_node, walk_length):
    output_path = open(os.path.join(data_path, "data.txt"), "w", encoding='utf8')

    hg, author_names, conf_names, paper_names = construct_graph(data_path)

    for conf_idx in tqdm.trange(hg.number_of_nodes('conf')):
        traces, _ = dgl.sampling.random_walk(
            hg, [conf_idx] * num_walks_per_node, metapath=['cp', 'pa', 'ap', 'pc'] * walk_length)
        for tr in traces:
            outline = ' '.join(
                (conf_names if i % 4 == 0 else author_names)[tr[i]]
                for i in range(0, len(tr), 2))  # skip paper
            print(outline, file=output_path)
    output_path.close()