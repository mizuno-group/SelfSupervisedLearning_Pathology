# -*- coding: utf-8 -*-
"""
# barlowtwins with recursive resuming

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/workspace/tggate'

# packages installed in the current environment
import sys
import os
import gc
import datetime
import argparse
import time
import random

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset

# original packages in src
sys.path.append(f"{PROJECT_PATH}/src/SelfSupervisedLearningPathology")
from sslmodel import data_handler as dh
import sslmodel.sslutils as sslutils
from mil.Attention import AttentionMultiWSI
import settings

# argument
parser = argparse.ArgumentParser(description='CLI learning')
# base settings
parser.add_argument('--note', type=str, help='barlowtwins running')
parser.add_argument('--seed', type=int, default=0)
# data settings
parser.add_argument('--fold', type=int, default=0) # number of fold
parser.add_argument('--target', type=str, help='target finding name') # number of fold
parser.add_argument('--layer', type=int, default=5) 
parser.add_argument('--dir_feature', type=str, help='input')
parser.add_argument('--dir_result', type=str, help='output')
# model/learning settings
parser.add_argument('--num_epoch', type=int, default=50) # epoch
parser.add_argument('--batch_size', type=int, default=64) # batch size
parser.add_argument('--lr', type=float, default=0.01) # learning rate
parser.add_argument('--patience', type=int, default=3) # early stopping
parser.add_argument('--delta', type=float, default=0.002) # early stopping

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# prepare data
class Dataset_WSI(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                folder:str="",
                target:str="",
                fold:int=None,
                layer:int=5,
                test:bool=True,
                ):
        # load data
        df_info=pd.read_csv(settings.file_classification)
        df_info["INDEX"]=list(range(df_info.shape[0]))
        if test:
            df_info=df_info[df_info["FOLD"]==fold]
        else:
            df_info=df_info[df_info["FOLD"]!=fold]
        self.lst_filein=[f"{folder}/fold{fold}_{i}_layer{layer}" for i in df_info["INDEX"]]
        self.labels=torch.Tensor(df_info[target].tolist()).reshape(-1,1)
        self.datanum = len(self.lst_filein)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = torch.Tensor(np.load(self.lst_filein[idx]).astype(np.float32))
        out_label = self.labels[idx]
        return out_data, out_label

def prepare_data(folder:str="", target:str="", layer:int=5, fold:int=0, batch_size:int=32, ):
    """
    data preparation
    
    """
    def collate_fn(batch):
        """padding"""
        data, label=[], []
        for x, y in batch:
            data.append(torch.unsqueeze(x, dim=1))
            label.append(y)
        data = rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        label=torch.stack(label, dim=0)
        return data, label
    # data
    train_dataset = Dataset_WSI(
        folder=folder,
        target=target,
        layer=layer,
        fold=fold,
        test=False,
        )
    test_dataset = Dataset_WSI(
        folder=folder,
        target=target,
        layer=layer,
        fold=fold,
        test=True,
    )
    # to loader
    train_loader = dh.prep_dataloader(
        train_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False,
        collate_fn=collate_fn,
        )
    test_loader = dh.prep_dataloader(
        test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False,
        collate_fn=collate_fn,
        )
    return train_loader, test_loader

def prepare_model(patience:int=7, delta:float=0, lr:float=0.003, num_epoch:int=150):
    """
    preparation of models
    Parameters
    ----------
        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """
    # model building with indicated name
    model=AttentionMultiWSI(n_features=512, hidden_layer=128, n_labels=1, attention_branches=1, label_smoothing=0.01)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    early_stopping = sslmodel.utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model, optimizer, scheduler, early_stopping

# train epoch
def train_epoch(model, train_loader, optimizer, epoch):
    """
    train for epoch
    with minibatch
    """
    # train
    model.train() # training
    train_batch_loss = []
    for x, label in train_loader:
        x = x.to(DEVICE)
        label = label.to(DEVICE)
        loss = model.calc_loss(x, label)
        train_batch_loss.append(loss.item())
        optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagation
        optimizer.step() # update parameters
    gc.collect()
    return model, np.mean(train_batch_loss)

# evaluate
def test_epoch(model, est_loader, optimizer, epoch):
    """
    train for epoch
    with minibatch
    """
    # train
    model.train() # training
    test_batch_loss = []
    for x, label in test_loader:
        loss = model.calc_loss(x, label)
        test_batch_loss.append(loss.item())
        optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagation
        optimizer.step() # update parameters
    gc.collect()
    return model, np.mean(test_batch_loss)

# train
def train(model, train_loader, test_loader, optimizer, scheduler, early_stopping, num_epoch:int=100, ):
    """ train ssl model """
    # settings
    start = time.time() # for time stamp
    train_loss=list()
    for epoch in range(num_epoch):
        # train
        model, train_epoch_loss = train_epoch(model, train_loader, optimizer, epoch)
        scheduler.step()
        train_loss.append(train_epoch_loss)
        # Log
        LOGGER.logger.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}'
            )
        LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
        # save model
        state = {
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "early_stopping":early_stopping,
            "train_loss":train_loss
        }
        torch.save(state, f'{DIR_NAME}/state.pt')
        # early stopping
        early_stopping(train_epoch_loss, model)
        if early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            return model, train_loss, True
    return model, train_loss, True

def main(resume=False):
    # 1. Preparing
    model, optimizer, scheduler, early_stopping = prepare_model(
        patience=args.patience, delta=args.delta, lr=args.lr, num_epoch=args.num_epoch
    )
    train_loader, test_loader = prepare_data(
        folder=args.dir_feature, target=args.target, layer=args.layer, fold=args.fold, batch_size=args.batch_size, 
    )
    # 2. Training
    model, train_loss, flag_finish = train(
        model, train_loader, test_loader, 
        optimizer, scheduler, early_stopping, 
        num_epoch=args.num_epoch, 
    )        
    # 3. save results & config
    if flag_finish:
        sslmodel.plot.plot_progress_train(train_loss, DIR_NAME)
        sslmodel.utils.summarize_model(
            model,
            None,
            DIR_NAME, lst_name=['summary_ssl.txt', 'model_ssl.pt']
        )
        
        LOGGER.to_logger(name='argument', obj=args)
        LOGGER.to_logger(
            name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
        )
        LOGGER.to_logger(name='scheduler', obj=scheduler)
    else:
        LOGGER.logger.info('reached max epoch / train')

if __name__ == '__main__':
    # Settings
    filename = os.path.basename(__file__).split('.')[0]
    DIR_NAME = PROJECT_PATH + '/result/' +args.dir_result # for output
    file_log = f'{DIR_NAME}/logger.pkl'
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    # DIR
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    # LOG
    now = datetime.datetime.now().strftime('%H%M%S')
    LOGGER = sslmodel.utils.logger_save()
    LOGGER.init_logger(filename, DIR_NAME, now, level_console='debug') 
    # Main
    main(resume=False)
