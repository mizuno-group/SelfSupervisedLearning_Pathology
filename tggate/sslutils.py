# -*- coding: utf-8 -*-
"""
# SSL Method

@author: Katsuhisa MORITA
"""

import numpy as np
import torch
import torchvision

import sslmodel
from sslmodel.models import barlowtwins, simsiam, byol, swav

class BarlowTwins:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512):
        model = barlowtwins.BarlowTwins(backbone, head_size=[head_size, 512, 128])
        criterion = barlowtwins.BarlowTwinsLoss()
        model.to(DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        train_transform = ssl.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=True, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512):
        model = sslmodel.models.barlowtwins.BarlowTwins(backbone, head_size=[head_size, 512, 128])
        model.load_state_dict(torch.load(model_path))
        model = model.backbone
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        x1, x2 = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # put data on GPU
        z1, z2 = model(x1), model(x2)
        loss = criterion(z1, z2)
        return loss

class Byol:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512):
        """return ssl model"""
        model = byol.BYOL(
            backbone, image_size=224, 
            hidden_layer=-1, 
            projection_size = 256, projection_hidden_size = 512, 
            moving_average_decay = 0.99,
            DEVICE=self.DEVICE,
            )
        criterion = byol.loss_fn
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=True, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512):
        """return backbone model"""
        model = byol.BYOL(
            encoder, image_size=224, 
            hidden_layer=-1, 
            projection_size = 256, projection_hidden_size = 512,
            moving_average_decay = 0.99,
            DEVICE=self.DEVICE,
            )
        model.load_state_dict(torch.load(model_path))
        model = model.online_encoder.net
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        x1, x2 = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # put data on GPU
        online_pred_one, online_pred_two, target_proj_one, target_proj_two = model(x1=x1, x2=x2) # forward
        loss = (criterion(online_pred_one, target_proj_two) + criterion(online_pred_two, target_proj_one)).mean() * 0.5 # loss
        return loss

class SwaV:
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512):
        """return ssl model"""
        model = swav.SwaV(
            backbone, head_size=[head_size, 512, 128],
            n_prototypes=512,
        )
        criterion = swav.SwaVLoss()
        model.to(self.DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=True, multi=True,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512):
        """return backbone model"""
        model = swav.SwaV(
            backbone, head_size=[head_size, 512, 128],
            n_prototypes=512,
        )
        model.load_state_dict(torch.load(model_path))
        model = model.backbone
        model.to(self.DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        model.prototypes.normalize()
        multi_crops_features = [model(x.to(self.DEVICE)) for x in data]
        high_resolution_crops = multi_crops_features[:2]
        low_resolution_crops = multi_crops_features[2:]
        loss = criterion(high_resolution_crops, low_resolution_crops)
        return loss

class SimSiam:        
    def __init__(self, DEVICE="cpu"):
        self.DEVICE=DEVICE

    def prepare_model(self, backbone, head_size:int=512, DEVICE="cpu"):
        """return ssl model"""
        model= simsiam.SimSiam(
            backbone,
            head_size=head_size,
            dim=1024,
            pred_dim=512,)
        criterion = simsiam.NegativeCosineSimilarity()
        model.to(DEVICE)
        return model, criterion

    def prepare_transform(
        self,
        color_plob=0.8,
        blur_plob=0.2,
        solar_plob=0.
        ):
        """return transforms for ssl"""
        train_transform = sslmodel.utils.ssl_transform(
            color_plob=color_plob,
            blur_plob=blur_plob, 
            solar_plob=solar_plob,
            split=True, multi=False,
            )
        return train_transform

    def prepare_featurize_model(self, backbone, model_path:str="", head_size:int=512):
        """return backbone model"""
        model= simsiam.SimSiam(
            backbone,
            head_size=head_size,
            dim=1024,
            pred_dim=512,)
        model=model.encoder
        model.to(DEVICE)
        return model

    def calc_loss(self, model, data, criterion):
        x1, x2 = data[0].to(self.DEVICE), data[1].to(self.DEVICE) # put data on GPU
        p1, p2, z1, z2 = model(x1=x1, x2=x2) # forward
        loss = 0.5 * (criterion(z1, p2) + criterion(z2, p1))
        return loss