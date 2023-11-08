# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

prepare dataloader

@author: Katsuhisa Morita, tadahaya
"""
import gc
import time
from typing import Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, WeightedRandomSampler
from PIL import Image

# dataset
class MyDataset(torch.utils.data.Dataset):
    """ to create my dataset """
    def __init__(self, 
                split:str='train',
                root:str='path',
                transform=None):
        if type(transform)!=list:
            self.transform = [transform]
        else:
            self.transform = transform

        # load from project folder
        DATA = np.load(f'{root}/data/pathmnist.npz')
        self.data = DATA[f'{split}_images']
        self.label = DATA[f'{split}_labels']
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_label = self.label[idx].astype(int)
        out_data = Image.fromarray(out_data).convert("RGB")
        if self.transform:
            for t in self.transform:
                out_data = t(out_data)
        return out_data,out_label

class TGGATE_SSL_Dataset(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                info_df=pd.DataFrame(),
                dir_col_name:str="DIR",
                fold_col_name:str="FOLD",
                fold_lst:list=[0,],
                sample_col_name:str="SAMPLE",
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # extracted target fold
        self._info_df=[]
        for fold in fold_lst:
            self._info_df.append(info_df[info_df[fold_col_name]==fold])
        self._info_df=pd.concat(self._info_df, axis=0)
        # list of file dir
        dir_lst = self._info_df[dir_col_name].tolist()
        sample_lst_lst = self._info_df[sample_col_name].tolist()
        self.dir_lst = [[f"{dir_name}.npy", sample] for dir_name, sample_lst in zip(dir_lst, sample_lst_lst) for sample in sample_lst]
        self.datanum = len(self.dir_lst)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = np.load(self.dir_lst[idx][0])[self.dir_lst[idx][1]]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

class TGGATE_SSL_Dataset_Batch(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                batch_number:int=None,
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # load data
        with open(f"/work/ga97/share/tggates/batch/batch_{batch_number}.npy", 'rb') as f:
            self.data = np.load(f)
        self.datanum = len(self.data)
        gc.collect()

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prep_dataloader(
    dataset, batch_size:int, shuffle:bool=True, num_workers:int=4, pin_memory:bool=True, drop_last:bool=True, sampler=None
    ) -> torch.utils.data.DataLoader:
    """
    prepare train and test loader
    
    Parameters
    ----------
    dataset: torch.utils.data.Dataset
        prepared Dataset instance
    
    batch_size: int
        the batch size
    
    shuffle: bool
        whether data is shuffled or not

    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing
    
    """
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn,
        drop_last=drop_last,
        sampler=sampler,
        )
    return loader

class BalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset, n_frac = None, n_samples = None):
        avg = np.mean(dataset.labels, axis=0)
        avg[avg == 0] = 0.5
        avg[avg == 1] = 0.5
        self.avg = avg
        weights = (1 / (1 - avg + 1e-8)) * (1 - dataset.labels) + (
            1 / (avg + 1e-8)
        ) * dataset.labels
        weights = np.max(weights, axis=1)
        # weights = np.ones_like(dataset.labels[:,0])
        self.weights = weights
        if n_frac:
            super().__init__(weights, int(n_frac * len(dataset)))
        elif n_samples:
            super().__init__(weights, n_samples)

def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def prep_data(
    train_x, train_y, test_x, test_y, batch_size,
    transform=(None, None), shuffle=(True, False),
    num_workers=None, pin_memory=None
    ) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    prepare train and test loader from data
    combination of prep_dataset and prep_dataloader for model building
    
    Parameters
    ----------
    train_x, train_y, test_x, test_y: arrays
        arrays for training data, training labels, test data, and test labels
    
    batch_size: int
        the batch size

    transform: a tuple of transform functions
        transform functions for training and test, respectively
        each given as a list
    
    shuffle: (bool, bool)
        indicates shuffling training data and test data, respectively
    
    num_workers: int
        the number of threads or cores for computing
        should be greater than 2 for fast computing
    
    pin_memory: bool
        determines use of memory pinning
        should be True for fast computing    

    """
    train_dataset = prep_dataset(train_x, train_y, transform[0])
    test_dataset = prep_dataset(train_x, train_y, transform[1])
    train_loader = prep_dataloader(
        train_dataset, batch_size, shuffle[0], num_workers, pin_memory
        )
    test_loader = prep_dataloader(
        test_dataset, batch_size, shuffle[1], num_workers, pin_memory
        )
    return train_loader, test_loader

def resize_dataset_dir(dataset, size:int=256):
    """ data resize for small scaling """
    dataset.dir_lst = dataset.dir_lst[:size]
    dataset.datanum = size
    return dataset

def resize_dataset(dataset, size:int=256):
    """ data resize for small scaling """
    dataset.data = dataset.data[:size]
    dataset.datanum = size
    return dataset