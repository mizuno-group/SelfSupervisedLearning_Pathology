import random

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
try:
    from openslide import OpenSlide
    import cv2
except:
    print("openslide is not available")

def check_file(str_file):
    """check input file can be opened or can't"""
    try:
        temp = openslide.OpenSlide(str_file)
        print("OK")
    except:
        print("can't open")

def make_patch(filein:str="", patch_size:int=256, num_patch:int=512, random_state:int=24771, inside=False,):
    """extract patch from WSI"""
    # set seed
    random.seed(random_state)
    # load
    wsi = OpenSlide(filein)
    # get patch mask
    if inside:
        mask=get_mask_inside(wsi, patch_size=patch_size, )
    else:
        mask = get_patch_mask(wsi, patch_size=patch_size)
    mask_shape=mask.shape
    # extract / append
    lst_number=np.array(range(len(mask.flatten())))[mask.flatten()]
    lst_number=random.sample(list(lst_number), num_patch)
    res = []
    ap = res.append
    for number in lst_number:
        v_h, v_w = divmod(number, mask_shape[1])
        patch_image=wsi.read_region(
            location=(int(v_w*patch_size), int(v_h*patch_size)),
            level=0,
            size=(patch_size, patch_size))
        ap(np.array(patch_image, np.uint8)[:,:,:3])
    res=np.concatenate(res).astype(np.uint8)
    return res, lst_number

def get_patch_mask(image, patch_size, threshold=None,):
    """
    Parameters
    ----------
    iamge: openslide.OpenSlide
    patch_size: int
    threshold: float or None (OTSU)
    Returns
    -------
    mask: np.array(int)[wsi_height, wsi_width]
    """
    level = image.get_best_level_for_downsample(patch_size)
    downsample = image.level_downsamples[level]
    ratio = patch_size / downsample
    whole = image.read_region(location=(0,0), level=level,
        size = image.level_dimensions[level]).convert('HSV')
    whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
    whole = np.array(whole, dtype=np.uint8)
    saturation = whole[:,:,1]
    if threshold is None:
        threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU)
    mask = saturation > threshold
    return mask

def get_mask_inside(
    image, 
    patch_size:int=224,
    slice_min_patch:int=1000,
    adjacent_cells_slice=[(-1,0),(1,0),(0,-1),(0,1),],
    adjacent_cells_inside=[(-1,0),(1,0),(0,-1),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)],
    ):
    # load
    level = image.get_best_level_for_downsample(patch_size)
    downsample = image.level_downsamples[level]
    ratio = patch_size / downsample
    whole = image.read_region(location=(0,0), level=level,
        size = image.level_dimensions[level]).convert('HSV')
    whole = whole.resize((int(whole.width / ratio), int(whole.height / ratio)))
    whole = np.array(whole, dtype=np.uint8)
    # get mask
    saturation = whole[:,:,1] #saturation
    threshold, _ = cv2.threshold(saturation, 0, 255, cv2.THRESH_OTSU)
    mask = saturation > threshold
    # get slices > slice_min_patch
    left_mask = np.full((mask.shape[0]+1, mask.shape[1]+1), fill_value=False, dtype=bool)
    left_mask[:-1, :-1] = mask
    n_left_mask = np.sum(left_mask)
    slice_idx = np.full_like(mask, fill_value=-1, dtype=int)
    i_slice = 0
    while n_left_mask > 0:
        mask_i = np.where(left_mask)
        slice_left_indices = [(mask_i[0][0], mask_i[1][0])]
        while len(slice_left_indices) > 0:
            i, j = slice_left_indices.pop()
            if left_mask[i, j]:
                slice_idx[i, j] = i_slice
                for adj_i, adj_j in adjacent_cells_slice:
                    if left_mask[i+adj_i, j+adj_j]:
                        slice_left_indices.append((i+adj_i, j+adj_j))
                left_mask[i, j] = False
                n_left_mask -= 1
        slice_mask = slice_idx == i_slice
        if np.sum(slice_mask) < slice_min_patch:
            slice_idx[slice_mask] = -1
        else:
            i_slice += 1
    # re-difinition mask
    mask=slice_idx>-1
    # get inside mask
    mask_inside=np.zeros(mask.shape, dtype=bool)
    for i in range(1,mask.shape[0]-1):
        for v in range(1,mask.shape[1]-1):
            tf = mask[i][v]
            if tf:
                for adj in adjacent_cells_inside:
                    tf&=mask[i+adj[0]][v+adj[1]]
            mask_inside[i][v]=tf
    return mask_inside

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
