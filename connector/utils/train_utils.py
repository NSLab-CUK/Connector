import torch
import numpy as np
import random

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def calculate_accuracy(predicted_labels, actual_labels):
    predictions = predicted_labels.max(1)[1].type_as(actual_labels)

    results = predictions.eq(actual_labels).double().sum()

    return results / len(actual_labels)