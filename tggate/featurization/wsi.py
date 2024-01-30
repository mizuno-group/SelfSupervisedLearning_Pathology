# -*- coding: utf-8 -*-
"""
# featurize module

@author: Katsuhisa MORITA
"""
import argparse
import time
import os
import re
import sys
import datetime
from typing import List, Tuple, Union, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from PIL import ImageOps, Image
import cv2
from openslide import OpenSlide

sys.path.append("/workspace/tggate/src/SelfSupervisedLearningPathology")
import sslmodel
import sslmodel.sslutils as sslutils
from tggate.utils import get_mask_inside

# argument
parser = argparse.ArgumentParser(description='CLI inference')
parser.add_argument('--note', type=str, help='feature')
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_pool_patch', type=int, default=None)
parser.add_argument('--num_patch', type=int, default=None)
parser.add_argument('--patch_size', type=int, default=224)
parser.add_argument('--model_name', type=str, default='ResNet18') # architecture name
parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
parser.add_argument('--model_path', type=str, default='')
parser.add_argument('--dir_result', type=str, default='')
parser.add_argument('--result_name', type=str, default='foldx_')
parser.add_argument('--pretrained', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--tggate_all', action='store_true')
parser.add_argument('--eisai', action='store_true')
parser.add_argument('--shionogi', action='store_true')
parser.add_argument('--rat', action='store_true')

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# Featurize Class
class Featurize:
    def __init__(self, DEVICE="cpu", lst_size=[], ):
        self.DEVICE=DEVICE
        self.lst_size=lst_size
        self.out_all=[[] for size in self.lst_size]
        self.out_all_pool=[[] for size in self.lst_size]

    def extraction():
        return None

    def featurize(self, model, data_loader, ):
        # featurize
        with torch.inference_mode():
            for data in data_loader:
                data = data.to(self.DEVICE)
                outs = self.extraction(model, data)
                for i, out in enumerate(outs):
                    self.out_all[i].append(out)

    def pooling(self, num_pool_patch:int=200):
        """max pooling"""
        for i, out in enumerate(self.out_all):
            self.out_all_pool[i].append(
                np.max(out.reshape(-1, num_pool_patch, self.lst_size[i]),axis=1)
                )
        # reset output list
        self.out_all=[[] for size in self.lst_size]

    def save_outall(self, folder="", name=""):
        for i, out in enumerate(self.out_all):
            out=np.concatenate(out).astype(np.float32)
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)
        self.out_all=[[] for size in self.lst_size]

    def save_outpool(self, folder="", name=""):
        for i, out in enumerate(self.out_all_pool):
            out=np.concatenate(out).astype(np.float32)
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)

class ResNet18Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[64,64,128,256,512],
            )
    def extraction(self, model, x):
        x = model[0](x)# conv1
        x = model[1](x)# bn
        x = model[2](x)# relu
        x = model[3](x)# maxpool
        x1 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[4](x)# layer1
        x2 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[5](x)# layer2
        x3 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,128)
        x = model[6](x)# layer3
        x4 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,256)
        x = model[7](x)# layer4
        x5 = torch.flatten(model[8](x), 1).detach().cpu().numpy().reshape(-1,512)
        return x1, x2, x3, x4, x5

class DenseNet121Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[64,128,256,512,1024],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x = model[0][2](x)
        x = model[0][3](x)
        x1 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,64)
        x = model[0][4](x)
        x = model[0][5](x)
        x2 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,128)
        x = model[0][6](x)
        x = model[0][7](x)
        x3 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,256)
        x = model[0][8](x)
        x = model[0][9](x)
        x4 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,512)
        x = model[0][10](x)
        x = model[0][11](x)
        x5 = torch.flatten(model[2](model[1](x)), 1).detach().cpu().numpy().reshape(-1,1024)
        return x1, x2, x3, x4, x5

class EfficientNetB3Featurize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[24,32,48,136,1536],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,24)
        x = model[0][2](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,32)
        x = model[0][3](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,48)
        x = model[0][4](x)
        x = model[0][5](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,136)
        x = model[0][6](x)
        x = model[0][7](x)
        x = model[0][8](x)
        x5 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,1536)
        return x1, x2, x3, x4, x5

class ConvNextTinyFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[96,192,384,768],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,96)
        x = model[0][2](x)
        x = model[0][3](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,192)
        x = model[0][4](x)
        x = model[0][5](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,384)
        x = model[0][6](x)
        x = model[0][7](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,768)
        return x1, x2, x3, x4

class ConvNextTinyFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[96,192,384,768],
            )
    def extraction(self, model, x):
        x = model[0][0](x)
        x = model[0][1](x)
        x1 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,96)
        x = model[0][2](x)
        x = model[0][3](x)
        x2 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,192)
        x = model[0][4](x)
        x = model[0][5](x)
        x3 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,384)
        x = model[0][6](x)
        x = model[0][7](x)
        x4 = torch.flatten(model[1](x), 1).detach().cpu().numpy().reshape(-1,768)
        return x1, x2, x3, x4

class RegNetY16gfFeaturize(Featurize):
    def __init__(self, DEVICE="cpu"):
        super().__init__(
            DEVICE=DEVICE, 
            lst_size=[32,48,120,336,888],
            )
    def extraction(self, model, x):
        x = model[0](x)
        x1 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,32)
        x = model[1][0](x)
        x2 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,48)
        x = model[1][1](x)
        x3 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,120)
        x = model[1][2](x)
        x4 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,336)
        x = model[1][3](x)
        x5 = torch.flatten(model[2](x), 1).detach().cpu().numpy().reshape(-1,888)
        return x1, x2, x3, x4, x5

def _load_state_dict_dense(model, weights):
    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls (pretrained-models). This pattern is used
    # to find such keys.
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )
    for key in list(weights.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            weights[new_key] = weights[key]
            del weights[key]
    model.load_state_dict(weights)
    return model

# DataLoader
class DatasetWSI(torch.utils.data.Dataset):
    def __init__(
        self,
        filein:str="",
        transform=None,
        patch_size=224,
        num_patch=None, random_state=24771,
        ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # load data
        self.wsi = OpenSlide(filein)
        mask_inside=get_mask_inside(self.wsi, patch_size=patch_size)
        lst_number=np.array(range(len(mask_inside.flatten())))[mask_inside.flatten()]
        self.lst_location=[[int(patch_size*v) for v in divmod(number, mask_inside.shape[1])] for number in lst_number]
        if num_patch:
            random.seed(random_state)
            self.lst_location=random.sample(self.lst_location, num_patch)
        self.datanum = len(self.lst_location)
        self.patch_size=patch_size

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_data=self.wsi.read_region(
            location=tuple(self.lst_location[idx]),
            level=0,
            size=(self.patch_size, self.patch_size),
            )
        out_data = np.array(out_data, np.uint8)[:,:,:3]
        out_data = Image.fromarray(out_data).convert("RGB")
        if self._transform:
            for t in self._transform:
                out_data = t(out_data)
        return out_data

def prepare_dataset(filein:str="", patch_size:int=224, batch_size:int=32, num_patch=None,):
    """
    data preparation
    
    """
    # normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    data_transform = transforms.Compose([
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        normalize
    ])
    # data
    dataset = DatasetWSI(
        filein=filein,
        transform=data_transform,
        num_patch=num_patch,
        patch_size=patch_size,
        )
    # to loader
    data_loader = sslmodel.data_handler.prep_dataloader(
        dataset, batch_size=batch_size, 
        shuffle=False,
        drop_last=False)
    return data_loader, dataset.lst_location

## Featurize Methods
# name: [Model_Class, last_layer_size, Featurize_Class]
DICT_MODEL = {
    "EfficientNetB3": [torchvision.models.efficientnet_b3, 1536, EfficientNetB3Featurize],
    "ConvNextTiny": [torchvision.models.convnext_tiny, 768, ConvNextTinyFeaturize],
    "ResNet18": [torchvision.models.resnet18, 512, ResNet18Featurize],
    "RegNetY16gf": [torchvision.models.regnet_y_1_6gf, 888, RegNetY16gfFeaturize],
    "DenseNet121": [torchvision.models.densenet121, 1024, DenseNet121Featurize],
}
DICT_SSL={
    "barlowtwins":sslutils.BarlowTwins,
    "swav":sslutils.SwaV,
    "byol":sslutils.Byol,
    "simsiam":sslutils.SimSiam,
    "wsl":sslutils.WSL,
}

def prepare_model(model_name:str='ResNet18', ssl_name="barlowtwins",  model_path="", pretrained=False, DEVICE="cpu"):
    """
    preparation of models
    Parameters
    ----------
        modelname (str)
            model architecture name

    """
    # model building with indicated name
    if pretrained:
        if model_name=="DenseNet121":
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder = _load_state_dict_dense(encoder, torch.load(model_path))
            model = nn.Sequential(
                *list(encoder.children())[:-1],
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
                )
        else:
            encoder = DICT_MODEL[model_name][0](weights=None)
            encoder.load_state_dict(torch.load(model_path))
            model=nn.Sequential(*list(encoder.children())[:-1])
    else:
        encoder = DICT_MODEL[model_name][0](weights=None)
        size = DICT_MODEL[model_name][1]
        if model_name=="DenseNet121":
            backbone = nn.Sequential(
                *list(encoder.children())[:-1],
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
                )
        else:
            backbone = nn.Sequential(
                *list(encoder.children())[:-1],
                )
        ssl_class = DICT_SSL[ssl_name](DEVICE=DEVICE)
        model = ssl_class.prepare_featurize_model(
            backbone, model_path=model_path,
            head_size=size,
        )
    model.to(DEVICE)
    model.eval()
    return model

def featurize_layer(
    model_name="", ssl_name="", model_path="", pretrained=False,
    lst_filein=list(), lst_filename=list(), dir_result="",
    num_pool_patch=None, num_patch=None,
    batch_size=128, patch_size=224,
    DEVICE="cpu", ):
    """featurize module"""
    # load model
    model=prepare_model(
        model_name=model_name, 
        ssl_name=ssl_name,
        model_path=model_path,
        pretrained=pretrained, 
        DEVICE=DEVICE
    )
    extract_class = DICT_MODEL[model_name][2](DEVICE=DEVICE)
    # result dir
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    # featurize
    for filein, filename in zip(lst_filein, lst_filename):
        data_loader, lst_location=prepare_dataset(filein=filein, batch_size=batch_size, patch_size=patch_size, num_patch=num_patch)
        extract_class.featurize(model, data_loader)
        if num_pool_patch:
            extract_class.pooling(num_pool_patch=num_pool_patch)
            extract_class.save_outpool(folder=dir_result, name=filename)
        else:
            extract_class.save_outall(folder=dir_result, name=filename)
        pd.to_pickle(lst_location, f"{dir_result}/{filename}_location.pickle")
        
def main():
    # settings
    start = time.time() # for time stamp
    print(f"start: {start}")

    # 1. load file locations and names
    if args.tggate_all: 
        df_info=pd.read_csv(f"/workspace/tggate/data/tggate_info_ext.csv")
        lst_filein=df_info["DIR"].tolist()
        lst_filename=list(range(df_info.shape[0]))
    if args.eisai:
        df_info=pd.read_csv("/workspace/tggate/data/eisai_info.csv")
        lst_filein=df_info["FILE"].tolist()
        lst_filename=df_info["INDEX"].tolist()
    if args.shionogi:
        df_info=pd.read_csv("/workspace/tggate/data/shionogi_info.csv")
        lst_filein=df_info["FILE"].tolist()
        lst_filename=df_info["INDEX"].tolist()
    if args.rat:
        df_info=pd.read_csv("/workspace/tggate/data/our_info.csv")
        lst_filein=[f"/workspace/HDD2/Lab/Rat_DILI/raw/{i}.tif" for i in df_info["NAME"].tolist()]
        lst_filename=df_info["INDEX"].tolist()
    lst_filename=[f"{result_name}{i}" for i in lst_filename]
    if args.resume:
        lst_fileout=[f"{args.dir_result}/{i}_layer5.npy" for i in lst_filename]
        lst_tf=[not os.path.isfile(i) for i in lst_fileout]
        lst_filein=[i for i, v in zip(lst_filein, lst_tf) if v]
        lst_filename=[i for i, v in zip(lst_filename, lst_tf) if v]
    # 2. inference & save results
    featurize_layer(
        model_name=args.model_name, ssl_name=args.ssl_name, 
        model_path=args.model_path, pretrained=args.pretrained,
        lst_filein=lst_filein, lst_filename=lst_filename,
        dir_result=args.dir_result,
        num_pool_patch=args.num_pool_patch, num_patch=args.num_patch,
        batch_size=args.batch_size, patch_size=args.patch_size,
        DEVICE=DEVICE)
    print('elapsed_time: {:.2f} min'.format((time.time() - start)/60))        

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    main()