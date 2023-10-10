# -*- coding: utf-8 -*-
"""
# utils for evaluation

@author: Katsuhisa MORITA
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sklearn.metrics as metrics
from sklearn.manifold import TSNE, MDS
from sklearn.metrics import pairwise_distances

from inmoose.pycombat import pycombat_norm

def standardize(x_train, x_test=None, train_only=False):
    """ standardize / fillna with 0 """
    ss = StandardScaler()
    if train_only:
        x_train = ss.fit_transform(x_train)
        x_train[np.isnan(x_train)]=0
        return x_train
    else:
        ss.fit(x_train)
        x_train = ss.transform(x_train)
        x_train[np.isnan(x_train)]=0
        x_test = ss.transform(x_test)
        x_test[np.isnan(x_test)]=0
        return x_train, x_test

def standardize_ctrl(arr, ctrl=list(), std=False):
    myu=np.mean(arr[ctrl], axis=0)
    sigma=np.std(arr[ctrl], axis=0,ddof=1)
    if std:
        arr=(arr-myu)/sigma
    else:
        # [-1,0,1] standardize
        arr=arr-myu
        arr[arr>0]=(arr/np.max(arr, axis=0))[arr>0]
        arr[arr<0]=-(arr/np.min(arr, axis=0))[arr<0]
    arr[np.isnan(arr)]=0
    return arr
    
def pca(x_train, x_test=None, n_components=32, train_only=False):
    if train_only:
        model = PCA(n_components=n_components)
        return model.fit_transform(x_train) 
    else:
        model = PCA(n_components=n_components)
        model.fit(x_train)
        x_train, x_test = model.transform(x_train), model.transform(x_test)
        return x_train, x_test   

def compress(arr, size=5):
    arr = arr.reshape(-1, size, arr.shape[1])
    arr_all = np.concatenate([
        arr.max(axis=1), 
        arr.min(axis=1),
        arr.mean(axis=1),
        ], axis=1)
    return arr_all

def load_array(layer:int=10, folder="", name="", size=None, pretrained=False, n_model=5):
    if pretrained:
        lst_arr_x=[np.load(f"{folder}/pretrained_layer{layer}.npy")]
    else:
        lst_arr_x = [np.load(f"{folder}/model{model}{name}_layer{layer}.npy") for model in range(n_model)]
    if size:
        lst_arr_x = [compress(arr_x, size=size) for arr_x in lst_arr_x]
    return lst_arr_x

def load_array_preprocess(
    layer:int=10, folder="", name="", size=None, pretrained=False, n_model=5,
    convertz=False,
    compression=True, n_components=2, 
    meta_viz=False, concat=False,
    ):
    lst_arr_x=load_array(
        layer=layer, folder=folder, name=name,
        size=size, pretrained=pretrained, n_model=n_model
        )
    arr_embedding=None
    if compression:
        lst_arr_x=[standardize(arr_x, train_only=True) for arr_x in lst_arr_x]
        lst_arr_x=[pca(arr_x, train_only=True, n_components=n_components) for arr_x in lst_arr_x]
        lst_arr_x=[standardize(arr_x, train_only=True) for arr_x in lst_arr_x]
    elif convertz:
        lst_arr_x=[standardize(arr_x, train_only=True) for arr_x in lst_arr_x]
    if concat:
        arr_embedding=np.concatenate(lst_arr_x, axis=1)
        arr_embedding=standardize(arr_embedding, train_only=True)
        arr_embedding=pca(arr_embedding, train_only=True)
    elif meta_viz:
        arr_embedding=meta_vizualize(lst_arr_x, projection_method="PCA")[0]
        arr_embedding=standardize(arr_embedding, train_only=True)
    return lst_arr_x, arr_embedding

def load_array_preprocess_two(
    layer:int=10, folder1="", folder2="", name="", size=None, pretrained=False, n_model=5,
    convertz=False, z_ctrl=False, ctrl_pos=[], ctrl_pos2=[],
    combat=False, 
    compression=False, n_components=16,
    concat=False, meta_viz=False,
    ):
    lst_arr_x=load_array(
        layer=layer, folder=folder1, name=name,
        size=size, pretrained=pretrained, n_model=n_model
        )
    lst_arr_x2=load_array(
        layer=layer, folder=folder2, name=name,
        size=size, pretrained=pretrained, n_model=n_model
        )
    arr_embedding=None
    arr_embedding2=None
    if combat:
        lst_arr_all=[np.array(pycombat_norm(
            data=np.concatenate([arr_x, arr_x2], axis=0).T,
            batch=[0]*arr_x.shape[0]+[1]*arr_x2.shape[0],
            prior_par=False,
            )) for arr_x, arr_x2 in zip(lst_arr_x, lst_arr_x2)]
        lst_arr_x, lst_arr_x2=[(i.T)[:lst_arr_x[0].shape[0]] for i in lst_arr_all], [(i.T)[lst_arr_x[0].shape[0]:] for i in lst_arr_all]
        del lst_arr_all
    if convertz:
        if z_ctrl:
            lst_arr_x=[standardize_ctrl(arr_x, ctrl=ctrl_pos,) for arr_x in lst_arr_x]
            lst_arr_x2=[standardize_ctrl(arr_x, ctrl=ctrl_pos2) for arr_x in lst_arr_x2]
        else:
            lst_arr_all=[standardize(arr_x, arr_x2) for arr_x, arr_x2 in zip(lst_arr_x, lst_arr_x2)] 
            lst_arr_x, lst_arr_x2 = [i[0] for i in lst_arr_all], [i[1] for i in lst_arr_all]
    if compression:
        if z_ctrl:
            lst_arr_x=[standardize_ctrl(arr_x, ctrl=ctrl_pos) for arr_x in lst_arr_x]
            lst_arr_x2=[standardize_ctrl(arr_x, ctrl=ctrl_pos2) for arr_x in lst_arr_x2]
            lst_arr_all=[[arr_x, arr_x2] for arr_x, arr_x2 in zip(lst_arr_x, lst_arr_x2)]         
        else:
            lst_arr_all=[standardize(arr_x, arr_x2) for arr_x, arr_x2 in zip(lst_arr_x, lst_arr_x2)]
        lst_arr_all=[pca(arrs[0], arrs[1], n_components=n_components) for arrs in lst_arr_all]
        lst_arr_all=[standardize(arrs[0], arrs[1]) for arrs in lst_arr_all] 
        lst_arr_x, lst_arr_x2 = [i[0] for i in lst_arr_all], [i[1] for i in lst_arr_all]
        del lst_arr_all
    if concat:
        arr_embedding=np.concatenate(lst_arr_x, axis=1)
        arr_embedding2=np.concatenate(lst_arr_x2, axis=1)
        arr_embedding, arr_embedding2=standardize(arr_embedding, arr_embedding2)
        arr_embedding, arr_embedding2=pca(arr_embedding, arr_embedding2, n_components=2)
    elif meta_viz:
        arr_embeddings=meta_vizualize([
            np.concatenate([arr_x, arr_x2], axis=0)
            for arr_x, arr_x2 in zip(lst_arr_x, lst_arr_x2
            )], projection_method="PCA")[0]
        arr_embedding, arr_embedding2 = arr_embeddings[:lst_arr_x[0].shape[0]], arr_embeddings[lst_arr_x[0].shape[0]:]
        arr_embedding, arr_embedding2 = standardize(arr_embedding, arr_embedding2)
    return lst_arr_x, lst_arr_x2, arr_embedding, arr_embedding2

def multi_dataframe(df, coef:int=10):
    """augmentation"""
    if coef!=1:
        ind=df["INDEX"].tolist()
        df_y=[]
        df=df.T
        lst_col=df.columns
        for col in lst_col:
            df_y+=[df[col]]*coef
        df_y=pd.concat(df_y, axis=1).T
        ind_new=[]
        for i in ind:
            ind_new+=[int(i*coef+v) for v in range(coef)]
        df_y["INDEX"]=ind_new
        return df_y
    else:
        return df

def load_tggate(coef:int=1, filein="/workspace/230727_pharm/data/processed/tggate_info.csv", time="24 hr", lst_compounds=list()):
    df_info =pd.read_csv(filein)
    df_info["INDEX"]=list(range(df_info.shape[0]))
    df_info = df_info[
        (df_info["COMPOUND_NAME"].isin(lst_compounds))
        & ((df_info["DOSE_LEVEL"]=="High")|(df_info["DOSE_LEVEL"]=="Control"))
        & (df_info["SACRI_PERIOD"] == time)
    ]
    df_info = df_info.loc[:,["COMPOUND_NAME", "DOSE", "INDEX",]]
    df_info["COMPOUND_NAME"]=["vehicle" if dose==0 else name for name, dose in zip(df_info["COMPOUND_NAME"], df_info["DOSE"])]
    df_info=multi_dataframe(df_info, coef=coef)
    return df_info

def load_eisai(coef:int=1, conv_name=True, filein="/workspace/230727_pharm/data/processed/eisai_info.csv", time="24 hr"):
    dict_name={
        "Corn Oil":"vehicle",
        "Bromobenzene":"bromobenzene",
        "CCl4":"carbon tetrachloride",
        "Naphthyl isothiocyanate":"naphthyl isothiocyanate",
        "Methylcellulose":"vehicle",
        "Acetaminophen":"acetaminophen",
    } # Eisai datasetname → TG-GATE name * vehicle
    df_info_eisai = pd.read_csv(filein)
    if conv_name:
        df_info_eisai["COMPOUND_NAME"]=[dict_name.get(i, i) for i in df_info_eisai["COMPOUND_NAME"]]
    df_info_eisai=multi_dataframe(df_info_eisai, coef=coef)
    df_info_eisai=df_info_eisai.sort_values(by=["COMPOUND_NAME", "INDEX"])
    return df_info_eisai

def load_our(coef:int=1, filein="/workspace/231006_lab/data/our_info.csv", time="24 hr"):
    dict_name={"control":"vehicle",}
    df_info=pd.read_csv(filein)
    df_info["COMPOUND_NAME"]=[dict_name.get(i, i) for i in df_info["COMPOUND_NAME"]]
    df_info=df_info[df_info["SACRI_PERIOD"]==time]
    df_info=df_info.loc[:,["COMPOUND_NAME", "DOSE", "BY", "INDEX"]]
    df_info=multi_dataframe(df_info, coef=coef)
    return df_info

def calc_stats(y_true, y_pred, lst_compounds, le):
    """ Compounds Prediction """
    # Macro Indicators
    lst_res=[]
    df_res=pd.DataFrame(index=lst_compounds)
    for target in lst_compounds:
        i = list(le.classes_).index(target)
        auroc = metrics.roc_auc_score(y_true == i, y_pred[:, i])
        precision, recall, thresholds = metrics.precision_recall_curve(y_true == i, y_pred[:, i])
        aupr = metrics.auc(recall, precision)
        mAP = metrics.average_precision_score(y_true == i, y_pred[:, i])
        lst_res.append([auroc, aupr, mAP])
    df_res=pd.DataFrame(lst_res)
    df_res.columns=["AUROC","AUPR","mAP"]
    df_res=df_res.T
    df_res.columns=lst_compounds
    df_res["Macro Average"]=df_res.mean(axis=1)
    # Micro Indicators
    acc = np.mean(np.argmax(y_pred, axis=1) == y_true)
    ba = metrics.balanced_accuracy_score(y_true, np.argmax(y_pred, axis=1))
    auroc = metrics.roc_auc_score(y_true, y_pred, average="micro", multi_class="ovr")
    mAP = metrics.average_precision_score(y_true, y_pred, average="micro")
    return df_res.T, acc, ba, auroc, mAP

def meta_vizualize(visualizations, projection_method=None):
    '''
    Reference (modified from):
        https://github.com/rongstat/meta-visualization/blob/main/Python%20Code/meta_visualization.py
    
    Inputs:
       visualizations [list of numpy arrays]
        - list of numpy arrays of size n x 2 (columns being the two-dimensional coordinates of visualization)
        - each row should be matched to the same observation/sample across all elements in the list
       projection_method [str or None]
        - options are "tSNE" or "MDS" for reprojecting the meta-distance matrix into a meta-visualization
    
    Approach:
        1. Computes euclidean distance matrices from 2D embeddings
        2. Ensembles the euclidean distance according to eigenvector with largest eigenvalue
        3. Uses projection_method ["MDS", "tSNE"] to transform euclidean distance matrix to 2D embeddings
        
    Returns:
        meta_visualization [numpy array] - size nx2 two-dimensional visualization genereated using the meta-visualization approach
            - returned only if projection_method is not None
        meta_distances [numpy array] - size nxn pairwise euclidean distance matrix generated using meta-visualization approach
    '''
    # define number of samples
    n = visualizations[0].shape[0]
    K = len(visualizations)
    
    # Iterate and record distance matrices
    X_distance_matrix_list = []
    
    # compute pairwise distance matrix
    for X_embedded in visualizations:
        assert X_embedded.shape[0] == n, "All visualization arrays need to have the same number of rows"
        assert X_embedded.shape[1] == 2, "All visualization arrays need to have two columns"
        X_distance = pairwise_distances(X_embedded)
        X_distance_matrix_list.append(X_distance)
             
    # Compute weights for meta-visualization
    meta_distances = np.zeros((n,n))
    weights = np.zeros((n,K))
    for j in range(n):
        # fill in comparison matrix
        comp_mat = np.zeros((K,K))
        for i in range(K):
            for k in range(K):
                comp_mat[i,k] = np.sum(X_distance_matrix_list[k][:,j]*X_distance_matrix_list[i][:,j])/np.sqrt(np.sum(X_distance_matrix_list[k][:,j]**2))/np.sqrt(np.sum(X_distance_matrix_list[i][:,j]**2))
        # Eigenscore
        w, v = np.linalg.eig(comp_mat)
        weights[j,:] = np.abs(v[:,0])
    
        # Ensembles distance matrices
        matrix_norms = []
        for i in range(K):
            matrix_norms.append(X_distance_matrix_list[i][:,j]/np.sqrt(np.sum(X_distance_matrix_list[i][:,j]**2)))
        
        temp = np.zeros(matrix_norms[0].shape)
        for i in range(K):
            temp += matrix_norms[i]*weights[j,i]
            
        meta_distances[:,j] = temp
    
    meta_distances = np.nan_to_num((meta_distances+meta_distances.T)/2)
    
    # Re-project on 2D embedding space
    if projection_method is None:
        return(meta_distances)
    else:
        if projection_method == "tSNE":
            tsne = TSNE(n_components=2, metric="precomputed", verbose=0).fit(meta_distances)
            meta_visualization = tsne.embedding_
        elif projection_method == "MDS":
            mds = MDS(n_components=2, verbose=0, dissimilarity="precomputed").fit(meta_distances)
            meta_visualization = mds.embedding_
        elif projection_method == "PCA":
            meta_visualization = PCA(n_components=2).fit_transform(meta_distances)
        return(meta_visualization, meta_distances)

def pseudo_F(array, df, label):
    Total = np.sum((array[np.newaxis, df["INDEX"].tolist(), :] - array[df["INDEX"].tolist(), np.newaxis, :]) ** 2)
    n = len(array)
    W = 0
    for com in df[label].unique():
        array_temp = array[df[df[label]==com]["INDEX"].tolist()]
        W += np.sum((array_temp[np.newaxis, :, :] - array_temp[:, np.newaxis, :]) ** 2)
    k = len(df[label].unique())
    F = ((Total - W) / (k - 1)) / (W / (n - k))
    return F