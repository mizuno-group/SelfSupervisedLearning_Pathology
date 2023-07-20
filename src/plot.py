# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

visualization 

@author: tadahaya
"""
import matplotlib.pyplot as plt

def plot_curve(train_loss_list, valid_loss_list, dir_name):
    """ plot learning curve """
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.rcParams['font.size'] = 18
    ax.plot(range(len(train_loss_list)), train_loss_list, color='purple', label='train loss')
    ax.plot(range(len(valid_loss_list)), valid_loss_list, color='orange', label='test loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.set_title('train and test loss')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid()
    plt.savefig(dir_name + '/curve.png',bbox_inches="tight")