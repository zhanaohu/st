import os
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset
import numpy as np


def checkDir(path: str) -> bool:
    if os.path.exists(path):
        return True
    else:
        return False


def checkDirAndCreate(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)
        pass


def save_model(save_path: str, model: nn.Module, dataset: str, epochs: int):
    save_path = f'{save_path}/{dataset}/epoch={epochs}'
    if not checkDir(save_path): checkDirAndCreate(save_path)
    model_path = os.path.join(save_path, 'model.pkl')
    torch.save(model, model_path)
    print('model has been saved')
    pass


def load_model(save_path: str, dataset: str, epochs: int):
    save_path = f'{save_path}/{dataset}/epoch={epochs}'
    model_path = os.path.join(save_path, 'model.pkl')
    assert os.path.isfile(model_path), f'Found no model in the {model_path}'
    model = torch.load(model_path)

    return model


def save_metric(save_path: str, dataset: str, epochs: int, metric: dict, current_pred_len: int):
    save_path = f'{save_path}/{dataset}/epoch={epochs}/pred_len={current_pred_len}'
    if not checkDir(save_path):
        checkDirAndCreate(save_path)
        pass

    MSE = metric['MSE']
    MAE = metric['MAE']
    np.save(f'{save_path}/mse.npy', MSE)
    np.save(f'{save_path}/mae.npy', MAE)

    pass


def load_metric(save_path: str, dataset: str, epochs: int, current_pred_len: int):
    save_path = f'{save_path}/{dataset}/epoch={epochs}/pred_len={current_pred_len}'
    MSE = np.load(f'{save_path}/mse.npy', allow_pickle=True)
    MAE = np.load(f'{save_path}/mae.npy', allow_pickle=True)

    return MSE, MAE


class PretrainDataset(Dataset):
    def __init__(self, data, sigma, p=0.5, gen_model=None, multiplier=10, similarity=1.0, device="cuda"):
        super().__init__()
        self.data = data
        self.p = p
        self.sigma = sigma
        self.gen_model = gen_model
        self.multiplier = multiplier
        self.N, self.T, self.D = data.shape
        self.similarity = similarity
        self.device = device

    def __getitem__(self, item):
        here_data = self.data.reshape(self.N * self.T, self.D)
        here_data = nn.Linear(self.D, 512)(here_data)
        here_data = self.gen_model(here_data)
        here_data = here_data.reshape((self.N, self.T, self.D))
        noisy = torch.randn(self.N, self.T, self.D).to(self.device)
        data_aug1 = here_data[item % self.N].detach()
        here_data = here_data * self.similarity + (1.0 - self.similarity) * noisy
        data_aug2 = here_data[item % self.N].detach()

        return data_aug1, data_aug2

    def __len__(self):
        return self.data.size(0) * self.multiplier

    def transform(self, x):
        return self.jitter(self.shift(self.scale(x)))

    def weak(self, x):
        return self.jitter(self.scale(x))

    def strong(self, x):
        return self.shift(self.jitter(x))
        pass

    def jitter(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.shape) * self.sigma)

    def scale(self, x):
        if random.random() > self.p:
            return x
        return x * (torch.randn(x.size(-1)) * self.sigma + 1)

    def shift(self, x):
        if random.random() > self.p:
            return x
        return x + (torch.randn(x.size(-1)) * self.sigma)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    #
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]
