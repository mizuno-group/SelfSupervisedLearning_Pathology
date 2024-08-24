# -*- coding: utf-8 -*-
"""
# Image Visualizer

@author: Katsuhisa MORITA
"""

from .analyzer import Analyzer

class Visualizer():
    def __init__(self):
        self.Analyzer=Analyzer()
        self.ImageProcessor=ImageProcessor()

    def load_model(self, dir_featurize_model="model.pt", dir_classification_models="folder"):
        self.Analyzer.load_model(dir_featurize_model=dir_featurize_model, dir_classification_models=dir_classification_models)

    def analyze_image(self, filein="image", size=448):
        image, probabilities=self.Analyzer.analyze(filein, size=size)
        self.image=image
        self.probabilities=probabilities

    def plot_image(self, plot_proba=True, strategy="max", ):


    def print_probabilities(self):



if __name__=="__main__":
    dat=Visualizer()
    dat.load_model(dir_featurize_model="", dir_classification_models="")
    dat.analyze_image(filein="", size=448)
    dat.plot_image(sizse=448)
    dat.print_probabilites()