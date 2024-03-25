#%% Imports -------------------------------------------------------------------

import numpy as np
from joblib import Parallel, delayed 
from scipy.ndimage import distance_transform_edt
from skimage.morphology import (
    disk, binary_erosion, binary_dilation
    )

#%% Functions -----------------------------------------------------------------

def preprocess(hstack):
    
    # Convert to float32
    hstack = hstack.astype("float32")
    
    # Mean normalization
    for z in range(hstack.shape[1]):
        for t in range(hstack.shape[0]):
            hstack[t,z,...] /= np.mean(hstack[t,z,...])
            
    # Min. projection & 0 to 1 normalization
    hstack_proj = np.std(hstack, axis=1) # min or std 
    pMax = np.percentile(hstack_proj, 99.9)
    hstack_proj[hstack_proj > pMax] = pMax
    hstack_proj = (hstack_proj / pMax).astype("float32")
    
    return hstack_proj

# -----------------------------------------------------------------------------

def get_all(msk):
    
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, label in enumerate(labels):
        tmp = msk == label
        tmp = distance_transform_edt(tmp)
        pMax = np.percentile(tmp[tmp > 0], 99.9)
        tmp[tmp > pMax] = pMax
        tmp = (tmp / pMax)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0).astype("float32")  
    
    return edm

def get_outlines(msk):
    
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, label in enumerate(labels):
        tmp = msk == label
        tmp = tmp ^ binary_erosion(tmp)
        tmp = binary_dilation(tmp, footprint=disk(3))
        tmp = distance_transform_edt(tmp)
        pMax = np.percentile(tmp[tmp > 0], 99.9)
        tmp[tmp > pMax] = pMax
        tmp = (tmp / pMax)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0).astype("float32")
    
    return edm
    
# -----------------------------------------------------------------------------

def get_patches(arr, size, overlap):
    
    # Get dimensions
    if arr.ndim == 2: nT = 1; nY, nX = arr.shape 
    if arr.ndim == 3: nT, nY, nX = arr.shape
    
    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1, yPad2 = yPad // 2, (yPad + 1) // 2
    xPad1, xPad2 = xPad // 2, (xPad + 1) // 2
    
    # Pad array
    if arr.ndim == 2:
        arr_pad = np.pad(
            arr, ((yPad1, yPad2), (xPad1, xPad2)), mode='reflect') 
    if arr.ndim == 3:
        arr_pad = np.pad(
            arr, ((0, 0), (yPad1, yPad2), (xPad1, xPad2)), mode='reflect')         
    
    # Extract patches
    patches = []
    if arr.ndim == 2:
        for y0 in y0s:
            for x0 in x0s:
                patches.append(arr_pad[y0:y0 + size, x0:x0 + size])
    if arr.ndim == 3:
        for t in range(nT):
            for y0 in y0s:
                for x0 in x0s:
                    patches.append(arr_pad[t, y0:y0 + size, x0:x0 + size])
            
    return patches

# -----------------------------------------------------------------------------

def merge_patches(patches, shape, size, overlap):
    
    # Get dimensions 
    if len(shape) == 2: nT = 1; nY, nX = shape
    if len(shape) == 3: nT, nY, nX = shape
    nPatch = len(patches) // nT

    # Get variables
    y0s = np.arange(0, nY, size - overlap)
    x0s = np.arange(0, nX, size - overlap)
    yMax = y0s[-1] + size
    xMax = x0s[-1] + size
    yPad = yMax - nY
    xPad = xMax - nX
    yPad1 = yPad // 2
    xPad1 = xPad // 2

    # Merge patches
    def _merge_patches(patches):
        count = 0
        arr = np.full((2, nY + yPad, nX + xPad), np.nan)
        for i, y0 in enumerate(y0s):
            for j, x0 in enumerate(x0s):
                if i % 2 == j % 2:
                    arr[0, y0:y0 + size, x0:x0 + size] = patches[count]
                else:
                    arr[1, y0:y0 + size, x0:x0 + size] = patches[count]
                count += 1 
        arr = np.nanmean(arr, axis=0)
        arr = arr[yPad1:yPad1 + nY, xPad1:xPad1 + nX]
        return arr
        
    if len(shape) == 2:
        arr = _merge_patches(patches)

    if len(shape) == 3:
        patches = np.stack(patches).reshape(nT, nPatch, size, size)
        arr = Parallel(n_jobs=-1)(
            delayed(_merge_patches)(patches[t,...])
            for t in range(nT)
            )
        arr = np.stack(arr)
        
    return arr
 