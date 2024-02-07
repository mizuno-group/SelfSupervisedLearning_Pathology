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
import sklearn.metrics as metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler

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
parser.add_argument('--label_smoothing', type=float, default=None) # label_smoothing
parser.add_argument('--weight', type=float, default=None) # loss weight
parser.add_argument('--ratio', type=float, default=1.0) # under sampling ratio
parser.add_argument('--batch_size', type=int, default=64) # batch size
parser.add_argument('--lr', type=float, default=0.01) # learning rate
parser.add_argument('--patience', type=int, default=5) # early stopping
parser.add_argument('--delta', type=float, default=0.002) # early stopping

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# prepare data
class Dataset_WSI_UnderSample_Batch(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                lst_labels,
                lst_fold,
                folder:str="",
                fold:int=None,
                fold2:int=None,
                layer:int=5,
                ramdom_state:int=24771,
                ratio:float=None,
                ):
        # set load data
        lst_labels_fold=[lst_labels[i] for i, x in enumerate(lst_fold) if x==fold]
        # sampling
        if ratio:
            label0=[i for i, x in enumerate(lst_labels_fold) if x==0]
            label1=[i for i, x in enumerate(lst_labels_fold) if x==1]
            random.seed(random_state)
            label0 = random.sample(label0, int(len(label0)*ratio))
            random.seed(args.seed)
            lst_sample=label0+label1
        else:
            lst_sample=list(range(len(lst_labels_fold)))
        # load data
        success_load=False
        while not success_load:
            try:
                self.data=pd.read_pickle(f"{folder_output}/fold{fold}_fold{fold2}_layer{layer}.pickle")
                success_load=True
            except:
                success_load=False
                print("try to load again")
        self.data=[torch.Tensor(self.data[i]) for i in lst_sample]
        # set
        self.labels=torch.Tensor([lst_labels_fold[i] for i in lst_sample]).reshape(-1,1)
        self.datanum = len(lst_sample)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data = self.data[idx]
        out_label = self.labels[idx]
        return out_data, out_label

def load_labels(target:str=""):
    df_info=pd.read_csv(settings.file_classification)
    lst_labels=df_info[target].tolist()
    lst_fold=df_info["FOLD"].tolist()
    return lst_labels, lst_fold

def collate_fn(batch):
    """padding"""
    data, label=zip(*batch)
    data = rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
    label=torch.stack(label, dim=0)
    return data, label

def prepare_sampling_data(lst_labels, lst_fold, folder:str="", layer:int=5, fold:int=0, fold2:int=0, batch_size:int=32, random_state:int=24771, ratio:float=0.5):
    """
    data preparation
    
    """
    # data
    train_dataset = Dataset_WSI_UnderSample_Batch(
        lst_labels,
        lst_fold,
        folder=folder,
        layer=layer,
        fold=fold,
        fold2=fold2,
        random_state=random_state,
        ratio=ratio
        )
    # to loader
    train_loader = dh.prep_dataloader(
        train_dataset, batch_size=batch_size,
        shuffle=True, drop_last=True,
        collate_fn=collate_fn,
        )
    return train_loader

def prepare_test_data(lst_labels, lst_fold, folder:str="", layer:int=5, fold:int=0, batch_size:int=32, ):
    """
    data preparation
    
    """
    def collate_fn(batch):
        """padding"""
        data, label=zip(*batch)
        data = rnn.pad_sequence(data, batch_first=True, padding_value=0.0)
        label=torch.stack(label, dim=0)
        return data, label
    # Dataset
    test_dataset = Dataset_WSI_UnderSample_Batch(
        lst_labels,
        lst_fold,
        folder=folder,
        fold=fold,
        fold2=fold,
        layer=layer,
        ratio=None,
    )
    # to loader
    test_loader = dh.prep_dataloader(
        test_dataset, batch_size=batch_size,
        shuffle=False, drop_last=False,
        collate_fn=collate_fn,
        )
    return test_loader

def prepare_model(patience:int=7, delta:float=0, lr:float=0.003, num_epoch:int=150, label_smoothing:float=None, weight=None):
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
    if weight:
        weight=torch.tensor([1.0, weight]).to(DEVIDE) # Loss weight for findings (1)
    model=AttentionMultiWSI(
        n_features=512, hidden_layer=128, 
        n_labels=1, attention_branches=1, 
        label_smoothing=label_smoothing,
        weight=weight)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epoch, eta_min=0
        )
    early_stopping = sslmodel.utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model, optimizer, scheduler, early_stopping

# train epoch
def train_epoch(model, optimizer, epoch, lst_train_labels, lst_fold):
    """
    train for epoch
    with minibatch
    """
    # set batch
    lst_batch=list(range(5))
    lst_batch.remove(args.fold)
    random.seed(args.seed+epoch)
    random.shuffle(lst_batch)
    random.seed(args.seed)
    # train
    model.train() # training
    train_batch_loss = []
    for fold2 in lst_batch:
        train_loader=prepare_sampling_data(
            lst_train_labels, lst_fold,
            folder=args.folder, layer=args.layer, fold=args.fold, fold2=fold2,
            batch_size=args.batch_size, random_state=args.seed+epoch, 
            ratio=args.ratio
            )
        for x, label in train_loader:
            x = x.to(DEVICE)
            label = label.to(DEVICE)
            loss = model.calc_loss(x, label)
            train_batch_loss.append(loss.item())
            optimizer.zero_grad() # reset gradients
            loss.backward() # backpropagation
            optimizer.step() # update parameters
        del x, train_loader
        gc.collect()
    return model, np.mean(train_batch_loss)

# evaluate
def evaluate(model, test_loader):
    y_true, y_pred = test_epoch(model, test_loader)
    auroc, aupr, mAP, acc, ba = calc_stats(y_true, y_pred)
    return auroc, aupr, mAP, acc, ba

def calc_stats(y_true, y_pred):
    auroc = metrics.roc_auc_score(y_true, y_pred)
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred)
    aupr = metrics.auc(recall, precision)
    mAP = metrics.average_precision_score(y_true, y_pred)
    y_pred_temp=[np.rint(i) for i in y_pred]
    acc = metrics.accuracy_score(y_true, y_pred)
    ba = metrics.balanced_accuracy_score(y_true, y_pred)
    return auroc, aupr, mAP, acc, ba

def test_epoch(model, test_loader):
    """
    return test probability
    """
    # train
    model.eval() # training
    y_true=[]
    y_pred=[]
    with torch.inference_mode():
        for x, label in test_loader:
            x=x.to(DEVICE)
            pred = model.predict_proba(x)
            y_true.append(label.numpy())
            y_pred.append(pred)
    y_true=np.concatenate(y_true, axis=0)
    y_pred=np.concatenate(y_pred, axis=0)
    return y_true, y_pred

# train
def train(model, test_loader, optimizer, scheduler, early_stopping, num_epoch:int=100, ):
    """ train ssl model """
    # settings
    start = time.time() # for time stamp
    train_loss=list()
    res_test=list()
    # load
    lst_train_labels, lst_fold=load_train_labels(target=args.target)
    for epoch in range(num_epoch):
        model, train_epoch_loss = train_epoch(model, optimizer, epoch, lst_train_labels, lst_fold)
        scheduler.step()
        train_loss.append(train_epoch_loss)
        # Log
        if (epoch+1)%100==0 and epoch!=0:
            # inference
            test_loader = prepare_test_data(
                lst_train_labels, lst_fold, folder=args.dir_feature, layer=args.layer, fold=args.fold, batch_size=args.batch_size, 
            )
            auroc, aupr, mAP, acc, ba = evaluate(model, test_loader, )
            del test_loader
            gc.collect()
            # export
            res_test.append([auroc, aupr, mAP, acc, ba])
            LOGGER.logger.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
            LOGGER.logger.info(
                f'Epoch:{epoch + 1}, train_loss:{train_epoch_loss:.3f}, test_auroc:{auroc:.3f}, test_mAP:{mAP:.3f}'
                )
            # save model
            state = {
                "epoch":epoch,
                "model_state_dict":model.state_dict(),
                "train_loss":train_loss
            }
            torch.save(state, f'{DIR_NAME}/epoch{epoch}_state.pt')
        # early stopping
        early_stopping(train_epoch_loss, model)
        if early_stopping.early_stop:
            LOGGER.logger.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            return model, train_loss
    return model, train_loss

def main(resume=False):
    # 1. Preparing
    model, optimizer, scheduler, early_stopping = prepare_model(
        patience=args.patience, delta=args.delta, lr=args.lr, num_epoch=args.num_epoch, label_smoothihng=args.label_smoothing, weight=args.weight,
    )

    # 2. Training
    model, train_loss = train(
        model, test_loader, 
        optimizer, scheduler, early_stopping, 
        num_epoch=args.num_epoch, 
    )        
    # 3. save results & config
    sslmodel.plot.plot_progress_train(train_loss, DIR_NAME)
    sslmodel.utils.summarize_model(
        model,
        None,
        DIR_NAME, lst_name=['summary_ssl.txt', 'model_attention.pt']
    )
    LOGGER.to_logger(name='argument', obj=args)
    LOGGER.to_logger(
        name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
    )
    LOGGER.to_logger(name='scheduler', obj=scheduler)

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
