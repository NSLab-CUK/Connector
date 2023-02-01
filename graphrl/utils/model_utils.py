import torch
import numpy as np

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

def hadamard_product(x, y):
    ''' The function to calculate the hadamard product of two vectors
        argument x: the first vector(tensor)
        argument y: the second vector(tensor)
        output: the result of hadamard product of two vectors
    '''
    return x * y

def mean(x, y):
    ''' The function to calculate the average of two vectors
        argument x: the first vector(tensor)
        argument y: the second vector(tensor)
        output: the average of the two vectors
    '''
    return (x + y) / 2.0

def l1(x, y):
    ''' The function to calculate the L1 Norm of two vectors
        argument x: the first vector(tensor)
        argument y: the second vector(tensor)
        output: the value of L1 norm
    '''
    return np.abs(x - y)

def l2(x, y):
    ''' The function to calculate the L2 Norm of two vectors
        argument x: the first vector(tensor)
        argument y: the second vector(tensor)
        output: result of L2 norm
    '''
    return np.power(x - y, 2)

def dot(x, y):
    ''' The function to calculate dot product of two vectors
        argument x: the first vector(tensor)
        argument y: the second vector(tensor)
        output: dot product of x and y
    '''
    return np.dot(x, y)


def concat(x, y):
    ''' The function to concat two vectors
    '''
    return np.concatenate(x, y)

VECTOR_FUNCTIONS = {
    'l1': l1,
    'l2': l2,
    'concat': concat,
    'mean': mean,
    'hadamard': hadamard_product
}

def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]



def create_alias_table(area_ratio):
    """
    :param area_ratio: sum(area_ratio)=1
    :return: accept,alias
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                 (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept:
    :param alias:
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.random() * N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]

def preprocess_nxgraph(graph):
    node2idx = {}
    idx2node = []
    node_size = 0
    for node in graph.graph.nodes():
        node2idx[node] = node_size
        idx2node.append(node)
        node_size += 1
    return idx2node, node2idx


def partition_dict(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in vertices.items():
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_list(vertices, workers):
    batch_size = (len(vertices) - 1) // workers + 1
    part_list = []
    part = []
    count = 0
    for v1, nbs in enumerate(vertices):
        part.append((v1, nbs))
        count += 1
        if count % batch_size == 0:
            part_list.append(part)
            part = []
    if len(part) > 0:
        part_list.append(part)
    return part_list


def partition_num(num, workers):
    if num % workers == 0:
        return [num // workers] * workers
    else:
        return [num // workers] * workers + [num % workers]
        