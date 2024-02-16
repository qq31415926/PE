from torch.nn.functional import normalize 
import torch
def min_max_normalize(tensor, min_value=0, max_value=1):
    """
    Min-Max 归一化函数
    :param tensor: 待归一化的 tensor
    :param min_value: 归一化后的最小值
    :param max_value: 归一化后的最大值
    :return: 归一化后的 tensor
    """
    if isinstance(tensor, list):
        tensor = torch.tensor(tensor, dtype = torch.float32)
    min_val = tensor.min()
    max_val = tensor.max()

    normalized_tensor = min_value + (max_value - min_value) * (tensor - min_val) / (max_val - min_val)
    
    return normalized_tensor