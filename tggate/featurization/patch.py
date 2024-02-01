# -*- coding: utf-8 -*-
"""
# predict / pooling

@author: Katsuhisa MORITA
"""
# path setting
PROJECT_PATH = '/workspace/tggate'

# packages installed in the current environment
import sys
import datetime
import argparse
import time

import numpy as np
import pandas as pd
import torch

# original packages in src
sys.path.append(f"{PROJECT_PATH}/src/SelfSupervisedLearningPathology")
import settings
from tggate import featurize
import sslmodel

# argument
parser = argparse.ArgumentParser(description='CLI inference')
parser.add_argument('--note', type=str, help='feature')
parser.add_argument('--seed', type=int, default=24771)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_patch', type=int, default=200)
parser.add_argument('--model_name', type=str, default='ResNet18') # architecture name
parser.add_argument('--ssl_name', type=str, default='barlowtwins') # ssl architecture name
parser.add_argument('--dir_model', type=str, default='')
parser.add_argument('--result_name', type=str, default='')
parser.add_argument('--folder_name', type=str, default='')
parser.add_argument('--pretrained', action='store_true')

parser.add_argument('--tggate', action='store_true')
parser.add_argument('--tggate_all', action='store_true')
parser.add_argument('--eisai', action='store_true')
parser.add_argument('--shionogi', action='store_true')
parser.add_argument('--rat', action='store_true')

args = parser.parse_args()
sslmodel.utils.fix_seed(seed=args.seed, fix_gpu=True) # for seed control

# DataLoader
class Dataset_Batch(torch.utils.data.Dataset):
    """ load for each version """
    def __init__(self,
                filein:str="",
                transform=None,
                num_patch=200,
                ):
        # set transform
        if type(transform)!=list:
            self._transform = [transform]
        else:
            self._transform = transform
        # load data, select num_patch
        data = np.load(filein, mmap_mode="r")
        n_images=int(len(data)//200)
        data = data[[x for i in range(n_images) for x in [200*i+v for v in range(num_patch)]]].astype(np.uint8).copy()
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

def prepare_dataset_batch(filein:str="", batch_size:int=32, num_patch:int=200):
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
    dataset = Dataset_Batch(
        filein=filein,
        transform=data_transform,
        num_patch=num_patch,
        )
    # to loader
    data_loader = sslmodel.data_handler.prep_dataloader(
        dataset, batch_size, 
        shuffle=False,
        drop_last=False)
    return data_loader

# Featurize Class
class Featurize:
    def __init__(self, DEVICE="cpu", lst_size=[], ):
        self.DEVICE=DEVICE
        self.lst_size=lst_size
        self.out_all=[np.zeros((0,size), dtype=np.float32) for size in self.lst_size]
        self.out_all_pool=[np.zeros((0,3*size), dtype=np.float32) for size in self.lst_size]
    
    def extraction():
        return None

    def featurize(self, model, data_loader, ):
        # featurize
        with torch.inference_mode():
            for data in data_loader:
                data = data.to(self.DEVICE)
                outs = self.extraction(model, data)
                for i, out in enumerate(outs):
                    self.out_all[i] = np.concatenate([self.out_all[i], out])
        
    def pooling(self, num_patch:int=200, ):
        for i, out in enumerate(self.out_all):
            self.out_all_pool[i] = np.concatenate([
                self.out_all_pool[i],
                self._pooling_array(out, num_patch=num_patch, size=self.lst_size[i])
                ])
        # reset output list
        self.out_all=[np.zeros((0,size), dtype=np.float32) for size in self.lst_size]

    def save_outpool(self, folder="", name=""):
        for i, out in enumerate(self.out_all_pool):
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)

    def save_outall(self, folder="", name=""):
        for i, out in enumerate(self.out_all):
            np.save(f"{folder}/{name}_layer{i+1}.npy", out)

    def _pooling_array(self, out,num_patch=64, size=256):
        """ return max/min/mean pooling array with num_patch"""
        out = out.reshape(-1, num_patch, size)
        data_max=np.max(out,axis=1)
        data_min=np.min(out,axis=1)
        data_mean=np.mean(out,axis=1)
        data_all = np.concatenate([data_max, data_min, data_mean], axis=1).astype(np.float32)
        return data_all

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
    model, model_name="", ssl_name="",
    batch_size=128, lst_filein=list(), 
    folder_name="", result_name="", 
    DEVICE="cpu", num_patch=200,):
    try:
        extract_class = DICT_MODEL[model_name][2](DEVICE=DEVICE)
    except:
        print("indicated model name is not implemented")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    # featurize
    for filein in lst_filein:
        data_loader=prepare_dataset_batch(filein=filein, batch_size=batch_size, num_patch=num_patch)
        extract_class.featurize(model, data_loader)
        extract_class.pooling(num_patch=num_patch)
    extract_class.save_outpool(folder=folder_name, name=result_name)

def main():
    # settings
    start = time.time() # for time stamp
    print(f"start: {start}")
    # 1. model construction
    model = prepare_model(
        model_name=args.model_name, 
        ssl_name=args.ssl_name, 
        model_path=args.dir_model,
        pretrained=args.pretrained,
        DEVICE=DEVICE
        )
    ## file names
    if args.tggate:
        df_info=pd.read_csv(settings.file_tggate)
    if args.eisai:
        df_info=pd.read_csv(settings.file_eisai)
        df_info=df_info.sort_values(by=["INDEX"])
    if args.shionogi:
        folder="/workspace/HDD2/pharm/shionogi"
        df_info=pd.read_csv(settings.file_shionogi)
        df_info=df_info.sort_values(by=["INDEX"])
    if args.rat:
        df_info=pd.read_csv(settings.file_our)
        df_info=df_info.sort_values(by=["INDEX"])
    lst_filein=df_info["DIR_PATCH"].tolist()        

    # 2. inference & save results
    featurize_layer(
        model, model_name=args.model_name,
        batch_size=args.batch_size, lst_filein=lst_filein,
        folder_name=args.folder_name, result_name=args.result_name,
        DEVICE=DEVICE, num_patch=args.num_patch)
    print('elapsed_time: {:.2f} min'.format((time.time() - start)/60))        

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # get device
    main()