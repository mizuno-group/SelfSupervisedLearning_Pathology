# -*- coding: utf-8 -*-
"""
# Evaluation in Open TG-GATEs dataset
# finding/prognosis classification with KFold split (k=5)

@author: Katsuhisa MORITA
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE, MDS
from sklearn.linear_model import LogisticRegression

from evaluate import utils

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.size"] = 14

class FoldPred:
    def __init__(self):
        self.df_info=None
        self.arr_x=None
        self.y=None
        self.le=None

    def evaluate(
        self,
        folder=f"",
        name="",
        layer=0,
        pretrained=False,
        convertz=True,
        compression=False, n_components=16,
        params_lr=dict(),
        finding=False, prognosis=False, #prediction mode/multi label
        moa=False, compound_name=False, #prediction mode/multi class
        lst_features=list(),
        ):
        # load
        if finding:
            self.df_info=self._load_classification(lst_features=lst_features)
        if prognosis:
            self.df_info=self._load_prognosis(lst_features=lst_features)
        if moa:
            self.df_info=self._load_moa()
        if compound_name:
            self.df_info=self._load_compound_name()

        # Predict / Evaluate
        lst_res=[]
        for fold in range(5):
            arr_x_train, arr_x_test = utils.load_array_fold(
                self.df_info, fold=fold, layer=layer,
                folder=folder, name=name, pretrained=pretrained,
                convertz=convertz,
                compression=compression, n_components=n_components,
            )
            y_train = self.df_info.loc[self.df_info["FOLD"]!=fold, self.lst_features].values
            y_test = self.df_info.loc[self.df_info["FOLD"]==fold, self.lst_features].values
            if finding or prognosis:
                y_pred = _predict_multilabel(arr_x_train, arr_x_test, y_train, params_lr)
                res = utils.calc_stats_multilabel(y_test, y_pred, self.lst_features)
            elif moa or compound_name:
                y_pred = _predict_multiclass(arr_x_train, arr_x_test, y_train, params_lr)
                res = utils.calc_stats_multiclass(y_test, y_pred, )
            lst_res.append(res)
        return res

    def _predict_multilabel(self, x_train, x_test, y_train, params):
        """prediction with logistic regression for multi label task"""
        y_pred_all=[]
        # for loop for one feature
        for i in range(y_train.shape[1]):
            lr = LogisticRegression(**params)
            lr.fit(x_train, y_train[:,i])
            y_pred = model.predict_proba(x_test)[:,[1]]
            y_pred_all.append(y_pred)
        y_pred_all=np.stack(y_pred_all).T
        return y_pred

    def _predict_multiclass(self, x_train, x_test, y_train, params):
        """prediction with logistic regression for one label, multi class task"""
        lr = LogisticRegression(**params)
        lr.fit(x_train, y_train)
        y_pred=lr.predict_proba(x_test)
        return y_pred

    def _load_classification(
        filein="/workspace/230310_tggate_liver/data/classification/finding.csv", 
        lst_features=list()):
        self.df_info=pd.read_csv(filein)
        self.df_info["INDEX"]=list(range(self.df_info.shape[0]))
        if lst_features:
            self.lst_features=lst_features
        else:
            self.lst_features=[
                'Degeneration, hydropic',
                'Degeneration, fatty',
                'Change, acidophilic',
                'Ground glass appearance',
                'Proliferation, oval cell',
                'Single cell necrosis',
                'Degeneration, granular, eosinophilic',
                'Swelling',
                'Increased mitosis',
                'Alteration, nuclear',
                'Change, basophilic',
                'Hypertrophy',
                'Necrosis',
                'Inclusion body, intracytoplasmic',
                'Proliferation, Kupffer cell',
                'Change, eosinophilic',
                'Proliferation, bile duct',
                'Microgranuloma',
                'Alteration, cytoplasmic',
                'Deposit, glycogen',
                'Hematopoiesis, extramedullary',
                'Fibrosis',
                'Cellular infiltration',
                'Vacuolization, cytoplasmic'
                ]

    def _load_prognosis(
        filein="/workspace/230310_tggate_liver/data/prognosis/finding.csv", 
        lst_features=list()):
        self.df_info=pd.read_csv(filein)
        self.df_info["INDEX"]=list(range(self.df_info.shape[0]))
        if lst_features:
            self.lst_features=lst_features
        else:
            self.lst_features=[
                'Granuloma',
                'Change, acidophilic',
                'Ground glass appearance',
                'Proliferation, oval cell',
                'Single cell necrosis',
                'Degeneration, granular, eosinophilic',
                'Swelling',
                'Cellular foci',
                'Increased mitosis',
                'Hypertrophy',
                'Necrosis',
                'Inclusion body, intracytoplasmic',
                'Deposit, pigment',
                'Proliferation, Kupffer cell',
                'Change, eosinophilic',
                'Proliferation, bile duct',
                'Microgranuloma',
                'Anisonucleosis',
                'Deposit, glycogen',
                'Hematopoiesis, extramedullary',
                'Fibrosis',
                'Cellular infiltration',
                'Vacuolization, cytoplasmic'
                ]

    def _load_moa(
        filein_all="/workspace/230310_tggate_liver/result/info_fold.csv",
        filein_moa="/workspace/230310_tggate_liver/data/processed/moa.csv",
        )
        # load metadata
        df_info = pd.read_csv(filein_all)
        df_info["INDEX"]=list(range(df_info.shape[0]))
        moa_df = pd.read_csv(filein_moa).rename(columns={"Unnamed: 0":"COMPOUND_NAME"})
        moa_df["MoA"] = np.argmax(moa_df[moa_df.columns[1:]].values,axis=1)
        df_info = pd.merge(df_info, moa_df, on = "COMPOUND_NAME")
        df_info = df_info[df_info["SACRI_PERIOD"].isin(["4 day", "8 day", "15 day", "29 day"]) & (moa_info["DOSE"]>0)]
        # set
        self.df_info = df_info.loc[:,["COMPOUND_NAME","FOLD","INDEX"]]
        self.lst_features="MoA"

    def _load_compound_name(
        filein_all="/workspace/230310_tggate_liver/result/info_fold.csv",
        filein_moa="/workspace/230310_tggate_liver/data/processed/moa.csv",
        )
        # load metadata
        df_info = pd.read_csv(filein_all)
        df_info["INDEX"]=list(range(df_info.shape[0]))
        moa_df = pd.read_csv(filein_moa).rename(columns={"Unnamed: 0":"COMPOUND_NAME"})
        moa_df["MoA"] = np.argmax(moa_df[moa_df.columns[1:]].values,axis=1)
        df_info = pd.merge(df_info, moa_df, on = "COMPOUND_NAME")
        df_info = df_info[df_info["SACRI_PERIOD"].isin(["4 day", "8 day", "15 day", "29 day"]) & (moa_info["DOSE"]>0)]
        # set
        self.df_info = df_info.loc[:,["MoA","FOLD","INDEX"]]
        self.lst_features="COMPOUND_NAME"