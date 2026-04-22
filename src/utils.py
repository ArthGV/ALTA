import numpy as np
import random
import torch

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class BatchGenerater(object):
    def __init__(self, path):
        feats = np.load(path, allow_pickle=True)
        self.feats= feats
        self.index = 0

    def has_next(self):
        if self.index < len(self.feats):
            return True
        return False

    def reset(self):
        self.index = 0
        random.shuffle(self.feats)

    def next_batch(self, batch_size):
        batch = self.feats[self.index : self.index + batch_size]
        self.index += batch_size
        batch_input = []
        for x, p, l in batch:
            x = torch.from_numpy(x)
            p = torch.from_numpy(np.array([p]))
            l = torch.from_numpy(l)
            batch_input.append([x, p, l])

        return batch_input
    
def format_output(output):
    temp_list = []
    for element in output:
        for sub_element in element:
            temp_list.append(sub_element)
    temp_array = np.array(temp_list)
    formated_output = temp_array.T
    return formated_output