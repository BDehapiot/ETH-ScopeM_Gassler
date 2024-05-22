#%% Imports -------------------------------------------------------------------

import cv2
import numpy as np
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed 
from scipy.ndimage import distance_transform_edt

# Skimage
from skimage.filters import gaussian
from skimage.draw import rectangle_perimeter
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import (
    disk, binary_erosion, binary_dilation, skeletonize, remove_small_objects
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
    for l, lab in enumerate(labels):
        tmp = msk == lab
        tmp = distance_transform_edt(tmp)
        pMax = np.percentile(tmp[tmp > 0], 99.9)
        tmp[tmp > pMax] = pMax
        tmp = (tmp / pMax)
        edm[l,...] = tmp
    edm = np.max(edm, axis=0).astype("float32")  
    
    return edm

# -----------------------------------------------------------------------------

def get_outlines(msk):
    
    labels = np.unique(msk)[1:]
    edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
    for l, lab in enumerate(labels):
        tmp = msk == lab
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

# -----------------------------------------------------------------------------

def predict(C1_proj, size, overlap):
    
    # Define model
    model = sm.Unet(
        'resnet34', 
        input_shape=(None, None, 1), 
        classes=1, 
        activation='sigmoid', 
        encoder_weights=None,
        )
    
    # Get patches
    patches = get_patches(C1_proj, size, overlap)
    patches = np.stack(patches)
    
    # Load weights & predict
    model.load_weights(Path(Path.cwd(), "model_weights_all.h5")) 
    predAll = model.predict(patches).squeeze()
    model.load_weights(Path(Path.cwd(), "model_weights_outlines.h5")) 
    predOut = model.predict(patches).squeeze()
    model.load_weights(Path(Path.cwd(), "model_weights_bodies.h5")) 
    predBod = model.predict(patches).squeeze()
    
    # Merge patches
    predAll = merge_patches(predAll, C1_proj.shape, size, overlap)
    predOut = merge_patches(predOut, C1_proj.shape, size, overlap)
    predBod = merge_patches(predBod, C1_proj.shape, size, overlap)
    
    return predAll, predOut, predBod

# -----------------------------------------------------------------------------

def process(
        C1_proj, C2_proj,
        predAll, predOut, predBod,
        threshAll, threshOut, threshBod,
        min_size, min_roundness
        ):
    
    # Get mask ----------------------------------------------------------------
    
    maskAll = gaussian(predAll, sigma=(0, 1, 1)) > threshAll # Parameter !!!
    maskOut = gaussian(predOut, sigma=(0, 1, 1)) > threshOut # Parameter !!!
    maskBod = gaussian(predBod, sigma=(0, 1, 1)) > threshBod # Parameter !!!
           
    # Process masks
    mask = []
    for t in range(C1_proj.shape[0]):
        
        mAll = maskAll[t, ...].copy()
        mOut = maskOut[t, ...]
        mBod = maskBod[t, ...]
        
        # Separate touching objects
        skel = skeletonize(mOut, method="lee")
        mAll[skel == 255] = 0
        mAll = remove_small_objects(mAll, min_size=min_size, connectivity=1)
        
        # Remove border objects (added feature)
        mAll = clear_border(mAll)
                
        # Filter masks 
        for prop in regionprops(label(mAll, connectivity=1)):
            idx = (prop.coords[:, 0], prop.coords[:, 1])
            roundness = 4 * np.pi * prop.area / (prop.perimeter ** 2)

            # Based on roundness
            if roundness < min_roundness:
                mAll[idx] = False
            
            # Not connected to body
            if not np.max(mBod[idx]): 
                mAll[idx] = False     
        
        # Append
        mask.append(mAll)  
    mask = np.stack(mask)
        
    # Get labels --------------------------------------------------------------

    labels = []
    labels.append(label(mask[0, ...], connectivity=1))
    for t in range(1, mask.shape[0]):
        
        labs = label(mask[t, ...], connectivity=1)
        
        # Track objects
        for prop in regionprops(labs) :
            idx = (prop.coords[:, 0], prop.coords[:, 1])
            val1 = labels[t-1][idx]
            val2 = val1[val1 != 0]
            val3, counts = np.unique(val2, return_counts=True)
            if val2.size > val1.size * 0.25: # Parameter !!!
                mode = val3[np.argmax(counts)]
            else:
                mode = 0
            labs[idx] = mode
            
        # Append
        labels.append(labs)
    labels = np.stack(labels)
    
    # Update mask
    mask = labels > 0
    
    # Get skel_labels 
    skel = np.zeros_like(mask)
    for t in range(labels.shape[0]):
        skel[t, ...] = skeletonize(binary_erosion(mask[t, ...]))
    skel_labels = labels.copy()
    skel_labels[skel == 0] = 0

    # Get data ----------------------------------------------------------------

    data = []
    display = np.zeros_like(labels)
    display += skel * 96 # Draw object skeleton
    for lab in np.unique(labels)[1:]:
        area = np.full(labels.shape[0], np.nan)
        length = np.full(labels.shape[0], np.nan)
        roundness = np.full(labels.shape[0], np.nan)
        intensity = np.full(labels.shape[0], np.nan)
        
        for t in range(labels.shape[0]):
            labs = labels[t, ...]     
            skel_labs = skel_labels[t, ...]
            
            for prop1 in regionprops(labs, intensity_image=C2_proj[t,...]):
                if prop1.label == lab:
                    
                    # Area, intensity & roundness
                    area[t] = prop1.area
                    roundness[t] = 4 * np.pi * prop1.area / (prop1.perimeter ** 2)
                    intensity[t] = np.sum(prop1.image_intensity)
                    
                    # Length
                    tmp_length = 0
                    tmp_lab = label(skel_labs == lab, connectivity=1)
                    for prop2 in regionprops(tmp_lab):
                        tmp_length += prop2.area - 1
                    tmp_length += (np.max(tmp_lab) - 1) * np.sqrt(2)
                    length[t] = tmp_length
                    
                    # Draw object squares
                    idx = np.where(labs == lab)
                    x0, y0 = np.min(idx[0]), np.min(idx[1])
                    x1, y1 = np.max(idx[0]), np.max(idx[1])
                    rr, cc = rectangle_perimeter(
                        (x0, y0), (x1, y1), shape=display[t, ...].shape)
                    display[t, rr, cc] = 255
                                                        
                    # Draw object texts
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(
                        display[t,...], f"{lab:02d}", 
                        (y0 - 18, x0 + 6), # depend on resolution !!!
                        font, 0.33, 255, 1, cv2.LINE_AA
                        ) 
                    cv2.putText(
                        display[t,...], f"{area[t]:.0f}", 
                        (y0 - 2, x0 - 6), # depend on resolution !!!
                        font, 0.33, 64, 1, cv2.LINE_AA
                        ) 
                    cv2.putText(
                        display[t,...], f"{roundness[t]:.2f}", 
                        (y0 - 2, x0 - 18), # depend on resolution !!!
                        font, 0.33, 64, 1, cv2.LINE_AA
                        ) 
                    cv2.putText(
                        display[t,...], f"{int(intensity[t])}", 
                        (y0 - 2, x0 - 30), # depend on resolution !!!
                        font, 0.33, 64, 1, cv2.LINE_AA
                        ) 
        
        # Append
        data.append({
            "label" : lab,
            "area" : area, 
            "length" : length,
            "roundness" : roundness, 
            "intensity" : intensity,
            })
    
    return mask, labels, skel, display, data 
 