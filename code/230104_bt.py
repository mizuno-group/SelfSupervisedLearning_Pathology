# -*- coding: utf-8 -*-
"""
# test version (small scale)

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/work/02/ga97/a97001/221125_medmnist_ssl'

# packages installed in the current environment
import sys

from symbol import parameters
sys.path.append(PROJECT_PATH)
import os
import datetime
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torchinfo import summary
import numpy as np
from tqdm import tqdm, trange
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

# original packages in src
from src import utils
from src import data_handler as dh
from src.models import resnet, barlowtwins, linearhead

# setup
now = datetime.datetime.now().strftime('%H%M%S')
file = os.path.basename(__file__).split('.')[0]
DIR_NAME = PROJECT_PATH + '/results/' + file + '_' + now # for output
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)
LOGGER = utils.init_logger(file, DIR_NAME, now, level_console='debug') # for logger
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
EXPANSION=1

# argument
parser = argparse.ArgumentParser(description='CLI template')
parser.add_argument('--note', type=str, help='note for this running')
parser.add_argument('--train', type=bool, default=True) # training or not
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--num_epoch', type=int, default=50) # epoch
parser.add_argument('--num_epoch_ssl', type=int, default=150) # epoch
parser.add_argument('--batch_size', type=int, default=128) # batch size
parser.add_argument('--batch_size_ssl', type=int, default=16) # batch size ssl
parser.add_argument('--lr', type=float, default=0.001) # learning rate
parser.add_argument('--lr_ssl', type=float, default=0.003) # learning rate for ssl
parser.add_argument('--modelname', type=str, default='ResNet18') # architecture name
parser.add_argument('--patience', type=int, default=5) # early stopping
parser.add_argument('--patience_ssl', type=int, default=3) # early stopping for ssl
parser.add_argument('--delta', type=float, default=0.) # early stopping
parser.add_argument('--delta_ssl', type=float, default=0.0) # early stopping for ssl

args = parser.parse_args()
utils.fix_seed(seed=args.seed, fix_gpu=False) # for seed control

# prepare data
def prepare_data():
    """
    data preparation
    
    """
    train_data_ssl_transform, train_data_transform, other_data_transform = utils.myrotation(split=True, multi=False)

    train_set_ssl = dh.ColonDataset(split='train', transform=train_data_ssl_transform, root=PROJECT_PATH)
    train_set = dh.ColonDataset(split='train', transform=train_data_transform, root=PROJECT_PATH)
    val_set = dh.ColonDataset(split='val', transform=other_data_transform, root=PROJECT_PATH)
    test_set = dh.ColonDataset(split='test', transform=other_data_transform, root=PROJECT_PATH)

    train_ssl_loader = dh.prep_dataloader(train_set_ssl, args.batch_size_ssl, num_workers=9, shuffle=True, drop_last=True)
    train_loader = dh.prep_dataloader(train_set, args.batch_size, num_workers=9, shuffle=True, drop_last=True)
    val_loader = dh.prep_dataloader(val_set, args.batch_size, num_workers=9, shuffle=False, drop_last=False)
    test_loader = dh.prep_dataloader(test_set, args.batch_size, num_workers=9, shuffle=False, drop_last=False)
    return train_ssl_loader, train_loader, val_loader, test_loader

# prepare model
def prepare_model(model, patience:int=7, delta:float=0., num_classes:int=9):
    """
    preparation of models
    Parameters
    ----------
        model
            self-supervised learned model

        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """

    # model building
    model_all = linearhead.LinearConversion(model.backbone, num_classes=num_classes, dim=512*EXPANSION) 
    model_all = utils.fix_params(model_all, all=False)
    model_all.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_all.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epoch, eta_min=0
        )
    early_stopping = utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model_all, criterion, optimizer, scheduler, early_stopping

def prepare_model_ssl(modelname:str='TestNet', patience:int=7, delta=0):
    """
    preparation of models
    Parameters
    ----------
        modelname (str)
            model architecture name
        
        patience (int)
            How long to wait after last time validation loss improved.

        delta (float)
            Minimum change in the monitored quantity to qualify as an improvement.

    """

    # model name settings
    dict_model = {
        "ResNet18": resnet.ResNet18,
        "ResNet34": resnet.ResNet34,
        "ResNet50": resnet.ResNet50,
    }
    # model building with indicated name
    try:
        encoder = dict_model[modelname]()
    except:
        print("indicated model name is not implemented")
        ValueError
        
    global EXPANSION
    EXPANSION = encoder.expansion
    backbone = nn.Sequential(*list(encoder.children())[:-1])
    model = barlowtwins.BarlowTwins(backbone, head_size=[512*EXPANSION, 512, 128])
    model.to(DEVICE)
    criterion = barlowtwins.BarlowTwinsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr_ssl)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epoch_ssl, eta_min=0
        )
    early_stopping = utils.EarlyStopping(patience=patience, delta=delta, path=f'{DIR_NAME}/checkpoint.pt')
    return model, criterion, optimizer, scheduler, early_stopping

# train epoch
def train_epoch_ssl(model, train_loader, criterion, optimizer):
    """
    train for epoch
    
    """
    model.train() # training
    train_batch_loss = []
    for data, _ in train_loader:
        x1, x2 = data[0].to(DEVICE), data[1].to(DEVICE) # put data on GPU
        z1, z2 = model(x1), model(x2)
        loss = criterion(z1, z2)
        train_batch_loss.append(loss.item())
        optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagation
        optimizer.step() # update parameters
    return model, np.mean(train_batch_loss)

def train_epoch(model, train_loader, val_loader, criterion, optimizer):
    """
    train for epoch
    
    """
    model.train() # training
    train_batch_loss = []
    for data, label in train_loader:
        data, label = data.to(DEVICE), label.to(DEVICE) # put data on GPU
        optimizer.zero_grad() # reset gradients
        output = model(data) # forward
        loss = criterion(output, label.squeeze_()) # calculate loss
        loss.backward() # backpropagation
        optimizer.step() # update parameters
        train_batch_loss.append(loss.item())
    model.eval() # test (validation)
    val_batch_loss = []
    with torch.no_grad():
        for data, label in val_loader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            loss = criterion(output, label.squeeze_())
            val_batch_loss.append(loss.item())
    return model, np.mean(train_batch_loss), np.mean(val_batch_loss)

# train
def train_ssl(model, train_loader, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100):
    """ train ssl model """
    train_loss=[]
    for epoch in trange(num_epoch):
        model, train_epoch_loss = train_epoch_ssl(model, train_loader, criterion, optimizer)
        scheduler.step()
        train_loss.append(train_epoch_loss)
        LOGGER.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}'
            )
        early_stopping(train_epoch_loss, model)
        if early_stopping.early_stop:
            LOGGER.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            break
    return model, train_loss

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epoch:int=100):
    """ train main model """
    train_loss = []
    val_loss = []
    for epoch in trange(num_epoch):
        model, train_epoch_loss, val_epoch_loss = train_epoch(
            model, train_loader, val_loader, criterion, optimizer
            )
        scheduler.step() # should be removed if not necessary
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
        LOGGER.info(
            f'Epoch: {epoch + 1}, train_loss: {train_epoch_loss:.4f}, val_loss: {val_epoch_loss:.4f}'
            )
        early_stopping(val_epoch_loss, model) # early stopping
        if early_stopping.early_stop:
            LOGGER.info(f'Early Stopping with Epoch: {epoch}')
            model.load_state_dict(torch.load(early_stopping.path))        
            break
    return model, train_loss, val_loss

# predict
def predict(model, dataloader):
    """
    prediction
    
    """
    model.eval()
    y_true = torch.tensor([]).to(DEVICE)
    y_pred = torch.tensor([]).to(DEVICE)
    with torch.no_grad():
        for data, label in dataloader:
            data, label = data.to(DEVICE), label.to(DEVICE)
            output = model(data)
            output = output.softmax(dim=-1) # pay attention: softmax function
            y_true = torch.cat((y_true, label), 0)
            y_pred = torch.cat((y_pred, output), 0)
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()
        auc, acc=evaluation(y_true, y_pred)
    return auc, acc

def evaluation(y_true, y_pred):
    """
    scoring module (following MedMNIST evaluation)
    Parameters
    ----------
        y_true
            the ground truth labels, shape: (n_samples,)
        
        y_pred
            the predicted score of each class, shape: (n_samples, n_classes)

    """
    # calculate auc/acc
    auc=0
    for label in range(y_pred.shape[1]):
        y_true_binary= (y_true==label).astype(float)
        y_pred_binary= y_pred[:, label]
        auc+=roc_auc_score(y_true_binary, y_pred_binary.squeeze())
    auc = auc/y_pred.shape[1]
    # calculate auc
    acc = accuracy_score(y_true, np.argmax(y_pred, axis=1).squeeze())
    return auc, acc

def main():
    if args.train:
        # training mode
        start = time.time() # for time stamp
        train_ssl_loader, train_loader, val_loader, test_loader = prepare_data()
        LOGGER.info(
            f'num_training_data: {train_loader.dataset.datanum}, num_val_data: {val_loader.dataset.datanum}, num_test_data: {test_loader.dataset.datanum}'
            )
        # 1. Self-Supervised Learning
        model, criterion, optimizer, scheduler, early_stopping = prepare_model_ssl(modelname=args.modelname, patience=args.patience_ssl, delta=args.delta_ssl)
        model, train_loss, = train_ssl(
            model, train_ssl_loader, criterion, optimizer, scheduler, early_stopping, num_epoch=args.num_epoch_ssl,
            )        
        utils.plot_progress_train(train_loss, DIR_NAME)
        utils.summarize_model(
            model, summary(model, x1=next(iter(train_ssl_loader))[0][0].size(), x2=next(iter(train_ssl_loader))[0][0].size()), DIR_NAME, lst_name=['summary_ssl.txt', 'model_ssl.pt']
            )
        # 2. learning
        model, criterion, optimizer, scheduler, early_stopping = prepare_model(model, patience=args.patience, delta=args.delta)
        model, train_loss, val_loss = train(
            model, train_loader, val_loader, criterion, optimizer, scheduler, early_stopping, num_epoch=args.num_epoch,
            )
        utils.plot_progress(train_loss, val_loss, DIR_NAME)
        utils.summarize_model(model, summary(model, next(iter(train_loader))[0].size()), DIR_NAME, lst_name=['summary.txt', 'model.pt'])
        # 3. evaluation
        auc, acc = predict(model, train_loader)
        LOGGER.info(f'train acc: {acc:.4f}, auc{auc: 4f}')
        auc, acc = predict(model, val_loader)
        LOGGER.info(f'val acc: {acc:.4f}, auc{auc: 4f}')
        auc, acc = predict(model, test_loader)
        LOGGER.info(f'test acc: {acc:.4f}, auc{auc: 4f}')
        # 4. save results & config
        utils.to_logger(LOGGER, name='argument', obj=args)
        utils.to_logger(LOGGER, name='loss', obj=criterion)
        utils.to_logger(
            LOGGER, name='optimizer', obj=optimizer, skip_keys={'state', 'param_groups'}
            )
        utils.to_logger(LOGGER, name='scheduler', obj=scheduler)
        LOGGER.info('elapsed_time: {:.2f} min'.format((time.time() - start)/60))
    else:
        # inference mode (not implemented)
        pass

if __name__ == '__main__':
    main()