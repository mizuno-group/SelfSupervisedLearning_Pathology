# -*- coding: utf-8 -*-
"""
# Image Analyzer

@author: Katsuhisa MORITA
"""
import time
import os
import re
import sys
import datetime
import random
from typing import List, Tuple, Union, Sequence

import numpy as np
import pandas as pd
from openslide import OpenSlide
from PIL import ImageOps, Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from .imageprocessor import ImageProcessor
from .model import FindingClassifier


class Analyzer:
    def __init__(self, DEVICE="cpu"):
        # analyzing class
        self.FindingClassifier=FindingClassifier(DEVICE=DEVICE)
        self.ImageProcessor=ImageProcessor()
        # data
        self.mask=None
        self.locations=None
        # result
        self.result_patch=None
        self.result_all=None

    def load_model(self, dir_featurize_model="model.pt", dir_classification_models="folder or ", style="dict"):
        self.FindingClassifier.load_featurize_model(
            dir_model=dir_featurize_model
        )
        self.FindingClassifier.load_classification_models(
            dir_models=dir_classification_models, style=style
        )

    def analyze(
        self, 
        filein, 
        batch_size=256, 
        patch_size=448, 
        model_patch_size=224,
        slice_min_patch=100, 
        ):
        self.mask = self.ImageProcessor.get_mask_inside(
            filein=filein, 
            patch_size=patch_size, 
            slice_min_patch=slice_min_patch
        )
        #self.image=self.ImageProcessor.load_patch(
        #    filein=filein, 
        #    mask=self.mask, 
        #    patch_size=patch_size,
        #    model_patch_size=model_patch_size
        #)
        self.locations=[
            self.ImageProcessor.get_locations(
                filein=filein, 
                mask=self.mask, 
                patch_size=patch_size,
                model_patch_size=model_patch_size
            ),# small size
            self.ImageProcessor.get_locations(
                filein=filein, 
                mask=self.mask, 
                patch_size=patch_size,
                model_patch_size=patch_size
            ),# large size
        ]
        data_loaders=[
            prepare_dataset_location(
                filein=filein,
                locations=self.locations[0], 
                batch_size=batch_size,
                patch_size=model_patch_size
            ),
            prepare_dataset_location(
                filein=filein,
                locations=self.locations[1], 
                batch_size=batch_size,
                patch_size=patch_size
            ),            
        ]
        self.result_patch, self.result_all =self.FindingClassifier.classify(
            data_loaders, num_pool=int(patch_size/model_patch_size)**2
        )

# DataLoader
class PatchDatasetData(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                data=None,
                transform=None,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # set
        self.data=data
        self.datanum = len(self.data)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_dataset_data(data=None, batch_size:int=128):
    """
    data preparation
    """
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.Resize((224,224), antialias=True),
        transforms.ToTensor(),
        normalize
    ])
    # data
    dataset = PatchDataset(
        data=data,
        transform=data_transform,
        num_patch=num_patch,
        )
    # to loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        sampler=None,
        collate_fn=None
        )
    return data_loader

class PatchDatasetLocation(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                filein="",
                locations=None,
                transform=None,
                patch_size=224,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # set
        self.wsi=OpenSlide(filein)
        self.locations=locations
        self.patch_size=patch_size
        self.datanum = len(self.locations)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data=self.wsi.read_region(
            location=self.locations[idx],
            level=0,
            size=(self.patch_size, self.patch_size)
        )
        out_data = Image.fromarray(
            np.array(out_data, np.uint8)[:,:,:3]
        ).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_dataset_location(filein="", locations=None, batch_size:int=128, patch_size=224,):
    """
    data preparation
    """
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.Resize((224,224), antialias=True),
        transforms.ToTensor(),
        normalize
    ])
    # data
    dataset = PatchDatasetLocation(
        filein=filein,
        locations=locations,
        transform=data_transform,
        patch_size=patch_size
        )
    # to loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
        drop_last=False,
        sampler=None,
        collate_fn=None
        )
    return data_loader

def _worker_init_fn(worker_id):
    """ fix the seed for each worker """
    np.random.seed(np.random.get_state()[1][0] + worker_id)