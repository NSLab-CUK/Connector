import torch
import numpy as np
import scipy.sparse as sp


def normalize_matrix(matrix):
    row_sum = np.array(matrix.sum(1))

    new_row = np.power(row_sum, -1).flatten()
    new_row[np.isinf(new_row)] = 0
    new_row = sp.diags(new_row)

    return new_row.dot(matrix)

def convert_sparse_matrix_to_sparse_tensor(matrix):
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((matrix.row, matrix.col)).astype(np.int64))
    tensor_matrix = torch.from_numpy(matrix.data)
    shape = torch.Size(matrix.shape)

    return torch.sparse.FloatTensor(indices, tensor_matrix, shape)