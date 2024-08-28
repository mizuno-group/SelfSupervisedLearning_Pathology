# -*- coding: utf-8 -*-
"""
# Image Visualizer

@author: Katsuhisa MORITA
"""
import gc
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from openslide import OpenSlide
import cv2

from .analyzer import Analyzer

class Visualizer():
    def __init__(self):
        self.Analyzer=Analyzer()
        self.ImageProcessor=ImageProcessor()
        self.filein=None

    def load_model(
        self, 
        dir_featurize_model="model.pt", 
        dir_classification_models="folder",
        style="dict"
        ):
        self.Analyzer.load_model(
            dir_featurize_model=dir_featurize_model, 
            dir_classification_models=dir_classification_models,
            style=style,    
        )

    def analyze(
        self, 
        filein="image", 
        batch_size=256,
        patch_size=448,
        model_patch_size=224,
        slice_min_patch=100,
        save_memory=True,
        ):
        self.Analyzer.analyze(
            filein=filein, 
            batch_size=batch_size,
            patch_size=patch_size,
            model_patch_size=model_patch_size,
            slice_min_patch=slice_min_patch,
        )
        # take results
        self.result_patch, self.result_all = self.Analyzer.result_patch, self.Analyzer.result_all
        self.mask=self.Analyzer.mask
        self.locations=self.Analyzer.locations[1]
        # set parameters
        self.filein=filein
        self.patch_size=patch_size
        if save_memory:
            del self.Analyzer
            gc.collect()
        self.image=image
        self.probabilities=probabilities

    def load_image(self, scale_factor=4, ):
        if not self.filein:
            print("Analyze before load image")
        else:
            image=OpenSlide(self.filein)
            level = image.get_best_level_for_downsample(self.patch_size/scale_factor)
            downsample = image.level_downsamples[level]
            ratio = patch_size/ scale_factor / downsample
            self.image=image.read_region(
                location=(0,0), 
                level=level,
                size = image.level_dimensions[level]
                ).convert('HSV')

    def plot_rawimage(self, ):

    def plot_anormaly(self, ):
        return

    def plot_highscore_patch(self,)

    def print_probabilities(self,):


if __name__=="__main__":
    dat=Visualizer()
    dat.load_model(dir_featurize_model="", dir_classification_models="")
    dat.analyze_image(filein="", size=448)
    dat.plot_image(sizse=448)
    dat.print_probabilites()