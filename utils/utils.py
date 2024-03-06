import os
import random
import pickle
import numpy as np
import torch
import torch.backends.cudnn as cudnn

def fix_seed(seed):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_extension(filename):
    if os.path.splitext(filename)[1] != ".pkl":
        return filename + ".pkl"
    return filename

def save_dataset(dataset, filename, display=True):
    filedir = os.path.split(filename)[0]
    if display:
        print(filename)
    if not os.path.isdir(filedir):
        os.makedirs(filedir)
    with open(check_extension(filename), 'wb') as f:
        pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

def load_dataset(filename):
    with open(check_extension(filename), 'rb') as f:
        return pickle.load(f)

def split_dataset(dataset, ratio=[8.0, 0.1, 0.1], random_seed=1234):
    assert abs(sum(ratio) - 1) < 1e-9, "sum of ratio should equal to 1."
    num_samples = len(dataset)
    split_size = []
    for i in range(len(ratio)):
        if i == len(ratio) - 1:
            split_size.append(num_samples - sum(split_size))
        else:
            split_size.append(int(num_samples * ratio[i]))
    print(f"split_size = {split_size}")
    return torch.utils.data.random_split(dataset, split_size, generator=torch.Generator().manual_seed(random_seed))

def set_device(gpu):
    """
    Parameters
    ----------
    gpu: int 
        Used GPU #. gpu=-1 indicates using cpu.

    Returns
    -------
    use_cuda: bool
        whether a gpu is used or not
    device: str
        device name
    """
    if gpu >= 0:
        assert torch.cuda.is_available(), "There is no available GPU."
        torch.cuda.set_device(gpu)
        device = f"cuda:{gpu}"
        use_cuda = True
        cudnn.benchmark = True
        print(f'selected device: GPU #{gpu}')
    else:
        device = "cpu"
        use_cuda = False
        print(f'selected device: CPU')
    return use_cuda, device

def calc_tour_length(tour, coords):
    tour_length = []
    for i in range(len(tour) - 1):
        path_length = np.linalg.norm(coords[tour[i]] - coords[tour[i + 1]])
        tour_length.append(path_length)
    tour_length = np.sum(tour_length)
    return tour_length

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def apply_along_axis(func, inputs, dim: int = 0):
    return torch.stack([
        func(input) for input in torch.unbind(inputs, dim=dim)
    ])

def batched_bincount(x, dim, max_value):
    target = torch.zeros(x.shape[0], max_value, dtype=x.dtype, device=x.device)
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target