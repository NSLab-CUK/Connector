import numpy as np

def calculate_node_degree(adj_matrix, node):
    degree = 0
    for neighbour in adj_matrix[node]:
        degree += int(adj_matrix[node][neighbour] != 0) 
    return degree

def encode_onehot(labels):
    classes = set(labels)
    class_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}

    labels_onehot = np.array(list(map(class_dict.get, labels)), dtype=np.int32)

    return labels_onehot