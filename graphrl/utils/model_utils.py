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