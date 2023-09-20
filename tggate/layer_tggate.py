# -*- coding: utf-8 -*-
"""
# predict / pooling

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/work/gd43/a97001'

# packages installed in the current environment
import sys
import os
import gc
import glob
import datetime
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from PIL import Image

# original packages in src
sys.path.append(PROJECT_PATH)
from src import ssl
from src.ssl import data_handler as dh
from src.ssl import utils
from src.ssl.models import barlowtwins
from src.tggate import featurize

# argument
parser = argparse.ArgumentParser(description='CLI inference')
parser.add_argument('--note', type=str, help='feature')
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--model_name', type=str, default='ResNet18') # architecture name
parser.add_argument('--dir_model', type=str, default='')
parser.add_argument('--result_name', type=str, default='')
parser.add_argument('--folder_name', type=str, default='')
parser.add_argument('--pretrained', action='store_true')

args = parser.parse_args()
utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

def main():
    # settings
    start = time.time() # for time stamp
    print(f"start: {start}")
    # 1. model construction
    model = featurize.prepare_model(
        model_name=args.model_name, model_path=args.dir_model,
        pretrained=args.pretrained, DEVICE=DEVICE
        )
    ## file names
    df_info=pd.read_csv(f"{PROJECT_PATH}/experiments_pharm/tggate_info.csv")
    lst_filein=[f"/work/gd43/share/tggates/liver/patch/ext/{i}.npy" for i in df_info["FILE"].tolist()]
    # 2. inference & save results
    featurize.featurize_layer(
        model, model_name=args.model_name,
        batch_size=args.batch_size, lst_filein=lst_filein,
        folder_name=args.folder_name, result_name=args.result_name,
        DEVICE=DEVICE, num_patch=NUM_PATCH)
    print('elapsed_time: {:.2f} min'.format((time.time() - start)/60))        

if __name__ == '__main__':
    NUM_PATCH=200
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    main()