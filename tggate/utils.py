import random

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from openslide import OpenSlide
from wsi_preprocess.saturation_otsu import get_patch_mask, get_slice_idx

def check_file(str_file):
    """check input file can be opened or can't"""
    try:
        temp = openslide.OpenSlide(str_file)
        print("OK")
    except:
        print("can't open")

def make_patch(filein:str="", patch_size:int=256, patch_number:int=1000, seed:int=24771):
    """extract patch from WSI"""
    # set seed
    random.seed(seed)
    # load
    wsi = OpenSlide(filein)
    # get patch mask
    mask = get_patch_mask(image_file=filein, patch_size=patch_size)
    mask_shape=mask.shape
    # extract / append
    lst_number=np.array(range(len(mask.flatten())))[mask.flatten()]
    lst_number=random.sample(list(lst_number), patch_number)
    res = []
    ap = res.append
    for number in lst_number:
        v_h, v_w = divmod(number, mask_shape[1])
        patch_image=wsi.read_region(
            location=(int(v_w*patch_size), int(v_h*patch_size)),
            level=0,
            size=(patch_size, patch_size))
        ap(np.array(patch_image, np.uint8)[:,:,:3])
    return res, lst_number

def make_groupkfold(group_col, n_splits:int=5):
    temp_arr = np.zeros((len(group_col),1))
    kfold = GroupKFold(n_splits = n_splits).split(temp_arr, groups=group_col)
    kfold_arr = np.zeros((len(group_col),1), dtype=int)
    for n, (tr_ind, val_ind) in enumerate(kfold):
        kfold_arr[val_ind]=int(n)
    return kfold_arr

def sampling_patch_from_wsi(patch_number:int=200, all_number:int=2000, len_df:int=0, seed:int=None):
    if seed is not None:
        random.seed(seed)
    random_lst = list(range(all_number))
    return [random.sample(random_lst, patch_number) for i in range(len_df)]
