# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

utilities

@author: Katsuhisa, tadahaya
"""
import os
import datetime
import random
import logging
from typing import List, Tuple, Union

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt
from sklearn import metrics
from PIL import ImageOps, Image
import torchvision.transforms as transforms
from torchinfo import summary

# assist model building
def fix_seed(seed:int=None,fix_gpu:bool=False):
    """ fix seed """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def fix_params(model, all=False):
    """ freeze model parameters """
    # freeze layers
    for param in model.parameters():
        param.requires_grad = False
    # except last layer
    if all:
        pass
    else:
        last_layer = list(model.children())[-1]
        for param in last_layer.parameters():
            param.requires_grad = True
    return model

def unfix_params(model):
    """ unfreeze model parameters """
    # activate layers 
    for param in model.parameters():
        param.requires_grad = True
    return model

class RandomRotate(object):
    """Implementation of random rotation.
    Randomly rotates an input image by a fixed angle. By default, we rotate
    the image by 90 degrees with a probability of 50%.
    This augmentation can be very useful for rotation invariant images such as
    in medical imaging or satellite imaginary.
    Attributes:
        prob:
            Probability with which image is rotated.
        angle:
            Angle by which the image is rotated. We recommend multiples of 90
            to prevent rasterization artifacts. If you pick numbers like
            90, 180, 270 the tensor will be rotated without introducing 
            any artifacts.
    
    """

    def __init__(self, prob: float = 0.5, angle: int = 90):
        self.prob = prob
        self.angle = angle

    def __call__(self, sample):
        """Rotates the images with a given probability.
        Args:
            sample:
                PIL image which will be rotated.
        
        Returns:
            Rotated image or original image.
        """
        prob = np.random.random_sample()
        if prob < self.prob:
            sample =  transforms.functional.rotate(sample, self.angle)
        return sample

def random_rotation_transform(
    rr_prob: float = 0.5,
    rr_degrees: Union[None, float, Tuple[float, float]] = 90,
    ) -> Union[RandomRotate, transforms.RandomApply]:
    if rr_degrees == 90:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=rr_degrees)
    else:
        # Random rotation with random angle defined by rr_degrees.
        return transforms.RandomApply([transforms.RandomRotation(degrees=rr_degrees)], p=rr_prob)

def myrotation(split=False, multi=False):
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # augmentation for ssl (consistent to ref)
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2
            ),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    if split:
        if multi:
            train_data_ssl_transform = MultiCropsTransform(augmentation)
        else:
            train_data_ssl_transform = TwoCropsTransform(augmentation)
    else:
        train_data_ssl_transform = augmentation

    # transformations
    train_data_transform = augmentation #230104~ changed
    other_data_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return train_data_ssl_transform, train_data_transform, other_data_transform

def myrotationDINO():
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    # augmentation for ssl (consistent to ref)
    augmentation = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
        transforms.RandomGrayscale(p=0.2),
    ])
    train_data_ssl_transform = DINOCropsTransform(augmentation)

    # transformations
    train_data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        random_rotation_transform(rr_prob=1., rr_degrees=[0,180]),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8
            ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([
            transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.2
            ),
        transforms.RandomResizedCrop((224, 224)),
        transforms.ToTensor(),
        normalize
    ]) #230104~ changed
    other_data_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    return train_data_ssl_transform, train_data_transform, other_data_transform

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

class MultiCropsTransform:
    """ return high and low resolution different crops from one image """
    def __init__(
        self, base_transform, 
        crop_counts=[2,6], 
        crop_sizes=[224,96]
        ):
        self.base_transform = base_transform
        self.crop_counts=crop_counts
        self.crop_sizes=crop_sizes
        
        # list of transforms for crop images
        self.crop_transforms = []
        for i in range(len(crop_sizes)):
            random_resized_crop = transforms.RandomResizedCrop(
                crop_sizes[i]
            )
            self.crop_transforms.extend([
                transforms.Compose([
                    random_resized_crop,
                    base_transform
                ])] * crop_counts[i]
            )

    def __call__(self, x):
        views = [crop_transform(x) for crop_transform in self.crop_transforms]
        return views

class DINOCropsTransform:
    """ return high and low resolution different crops from one image """
    def __init__(
        self, base_transform, 
        crop_counts=6, 
        crop_sizes=[224,96]
        ):
        self.base_transform = base_transform
        self.crop_counts=crop_counts
        self.crop_sizes=crop_sizes
        
        # list of transforms for crop images
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.crop_transforms = []
        global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(crop_sizes[0], scale=(0.4, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            base_transform,
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=1
                ),
            transforms.ToTensor(),
            normalize                
        ])  
        global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(crop_sizes[0], scale=(0.4, 1), interpolation=transforms.InterpolationMode.BICUBIC),
            base_transform,
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.1
                ),
            transforms.RandomSolarize(threshold=128, p=0.2),                
            transforms.ToTensor(),
            normalize                
        ])    
        local_transform = transforms.Compose([
            transforms.RandomResizedCrop(crop_sizes[1], scale=(0.05, 0.4), interpolation=transforms.InterpolationMode.BICUBIC),
            base_transform,
            transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))], p=0.5
                ),
            transforms.ToTensor(),
            normalize
        ])
        self.crop_transforms=[global_transform1, global_transform2]
        self.crop_transforms.extend([local_transform] * crop_counts)

    def __call__(self, x):
        views = [crop_transform(x) for crop_transform in self.crop_transforms]
        return views    

# logger
def init_logger(
    module_name:str, outdir:str='', tag:str='',
    level_console:str='warning', level_file:str='info'
    ):
    """
    initialize logger
    
    """
    level_dic = {
        'critical':logging.CRITICAL,
        'error':logging.ERROR,
        'warning':logging.WARNING,
        'info':logging.INFO,
        'debug':logging.DEBUG,
        'notset':logging.NOTSET
        }
    if len(tag)==0:
        tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(
        level=level_dic[level_file],
        filename=f'{outdir}/log_{tag}.txt',
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y%m%d-%H%M%S',
        )
    logger = logging.getLogger(module_name)
    sh = logging.StreamHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
        )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def to_logger(
    logger, name:str='', obj=None, skip_keys:set=set(), skip_hidden:bool=True
    ):
    """ add instance information to logging """
    logger.info(name)
    for k,v in vars(obj).items():
        if k not in skip_keys:
            if skip_hidden:
                if not k.startswith('_'):
                    logger.info('  {0}: {1}'.format(k,v))
            else:
                logger.info('  {0}: {1}'.format(k,v))

# learning tools
class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    add little changes from from https://github.com/Bjarten/early-stopping-pytorch/pytorchtools.py
    """
    def __init__(self, patience:int=7, delta:float=0, path:str='checkpoint.pt'):
        """
        Parameters
        ----------
            patience (int)
                How long to wait after last time validation loss improved.

            delta (float)
                Minimum change in the monitored quantity to qualify as an improvement.

            path (str): 
                Path for the checkpoint to be saved to.
   
        """
        self.patience = patience
        self.counter = 0
        self.best_score = np.Inf
        self.early_stop = False
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        if val_loss > self.best_score - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), self.path)

    def delete_checkpoint(self):
        os.remove(self.path)

# save & export
def summarize_model(model, summary, outdir, lst_name=['summary.txt', 'model.pt']):
    """
    summarize model using torchinfo

    Parameters
    ----------
    outdir: str
        output directory path

    model:
        pytorch model
    
    size:
        size of input tensor
    
    """
    try:
        with open(f'{outdir}/{lst_name[0]}', 'w') as writer:
            writer.write(repr(summary))
    except ModuleNotFoundError:
        print('!! CAUTION: no torchinfo and model summary was not saved !!')
    torch.save(model.state_dict(), f'{outdir}/{lst_name[1]}')


# plot
def plot_progress(train_loss, test_loss, outdir):
    """ plot learning progress """
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 18
    ax.plot(list(range(1, len(train_loss) + 1, 1)), train_loss, c='purple', label='train loss')
    ax.plot(list(range(1, len(test_loss) + 1, 1)), test_loss, c='orange', label='test loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + '/progress.tif', dpi=100, bbox_inches='tight')

def plot_progress_train(train_loss, outdir):
    """ plot learning progress """
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 18
    ax.plot(list(range(1, len(train_loss) + 1, 1)), train_loss, c='purple', label='train loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir + '/progress_train.tif', dpi=100, bbox_inches='tight')

def plot_accuracy(scores, labels, outdir):
    """ plot learning progress """
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auroc = metrics.auc(fpr, tpr)
    precision, _, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(tpr, precision)
    fig, axes = plt.subplots(1, 2, tight_layout=True)
    plt.rcParams['font.size'] = 18
    axes[0, 1].plot(fpr, tpr, c='purple')
    axes[0, 1].set_title(f'ROC curve (area: {auroc:.3})')
    axes[0, 1].set_xlabel('FPR')
    axes[0, 1].set_ylabel('TPR')
    axes[0, 2].plot(tpr, precision, c='orange')
    axes[0, 2].set_title(f'PR curve (area: {aupr:.3})')
    axes[0, 2].set_xlabel('Recall')
    axes[0, 2].set_ylabel('Precision')
    plt.grid()
    plt.savefig(outdir + '/accuracy.tif', dpi=100, bbox_inches='tight')
    df = pd.DataFrame({'labels':labels, 'predicts':scores})
    df.to_csv(outdir + '/predicted.txt', sep='\t')
    return auroc, aupr