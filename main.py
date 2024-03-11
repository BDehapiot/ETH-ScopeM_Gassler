#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 

# Functions
from functions import preprocess, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
loc_path = Path("D:/local_Gassler/data")
data_path = Path(Path.cwd(), "data")
model_all_path = Path(Path.cwd(), "model_weights_all.h5")
model_outlines_path = Path(Path.cwd(), "model_weights_outlines.h5")
model_bodies_path = Path(Path.cwd(), "model_weights_bodies.h5")
exp_name, exp_numb = "IND", "001"

# Patches
downscale_factor = 4
size = 512 // downscale_factor
overlap = size // 2

#%% Preprocessing -------------------------------------------------------------

C1 = io.imread(Path(loc_path, f"C1_{exp_name}_{exp_numb}.tif"))
C1_min = preprocess(C1)
patches = get_patches(C1_min, size, overlap)
patches = np.stack(patches)

#%% Predict -------------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )

# Load weights & predict
model.load_weights(model_all_path) 
predAll = model.predict(patches).squeeze()
model.load_weights(model_outlines_path) 
predOut = model.predict(patches).squeeze()
model.load_weights(model_bodies_path) 
predBod = model.predict(patches).squeeze()

# Merge patches
print("Merge patches :", end='')
t0 = time.time()
predAll = merge_patches(predAll, C1_min.shape, size, overlap)
predOut = merge_patches(predOut, C1_min.shape, size, overlap)
predBod = merge_patches(predBod, C1_min.shape, size, overlap)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# # Display
# viewer = napari.Viewer()
# viewer.add_image(C1_min,  blending="additive", opacity=0.33) 
# viewer.add_image(predAll, blending="additive", colormap="bop blue") 
# viewer.add_image(predOut, blending="additive", colormap="bop orange") 
# viewer.add_image(predBod, blending="additive", colormap="bop purple") 

#%%

from skimage.filters import gaussian
from skimage.draw import rectangle_perimeter
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, skeletonize, remove_small_objects

# # -----------------------------------------------------------------------------

# Parameters
sigma = (0, 1, 1)
threshAll = 0.05
threshOut = 0.25
threshBod = 0.05
min_size = 32

# # -----------------------------------------------------------------------------

# Get masks
maskAll = gaussian(predAll, sigma=sigma) > threshAll
maskOut = gaussian(predOut, sigma=sigma) > threshOut
maskBod = gaussian(predBod, sigma=sigma) > threshBod

# Process masks
pMaskAll, pMaskOut, pMaskBod = [], [], []
for t in range(C1_min.shape[0]):
    
    mAll = maskAll[t, ...].copy()
    mOut = maskOut[t, ...]
    mBod = maskBod[t, ...]
    
    # Separate touching objects
    skel = skeletonize(mOut, method="lee")
    mAll[skel == 255] = 0
    mAll = remove_small_objects(mAll, min_size=min_size, connectivity=1)
    
    # Filter masks 
    for prop in regionprops(label(mAll, connectivity=1)):
        idx = (prop.coords[:, 0], prop.coords[:, 1])
        roundness = 4 * np.pi * prop.area / (prop.perimeter ** 2)

        # Based on roundness
        if roundness < 0.5: # Parameter !!!
            mAll[idx] = False
        
        # Not connected to body
        if not np.max(mBod[idx]): 
            mAll[idx] = False
    
    # Append
    pMaskAll.append(mAll)    
pMaskAll = np.stack(pMaskAll)

# -----------------------------------------------------------------------------

# Get labels
labels = []
labels.append(label(pMaskAll[0, ...], connectivity=1))
for t in range(1, pMaskAll.shape[0]):
    
    labs = label(pMaskAll[t, ...], connectivity=1)
    med = np.median(np.stack(labels), axis=0)
    props = regionprops(labs)    
    
    # Track objects
    for prop in props:
        idx = (prop.coords[:, 0], prop.coords[:, 1])
        val1 = labels[t-1][idx]
        val2 = val1[val1 != 0]
        val3, counts = np.unique(val2, return_counts=True)
        if val2.size > val1.size * 0.5: 
            mode = val3[np.argmax(counts)]
        else:
            mode = 0
        labs[idx] = mode
    labels.append(labs)
labels = np.stack(labels)

# -----------------------------------------------------------------------------

display = np.zeros_like(labels)
for t in range(labels.shape[0]):
    labs = labels[t, ...]
    # labs = label(pMaskAll[t, ...])
    for lab in np.unique(labs):
        idx = np.where(labs == lab)
        start = (np.min(idx[0]), np.min(idx[1]))
        end = (np.max(idx[0]), np.max(idx[1]))
        rr, cc = rectangle_perimeter(start, end,
            shape=display[t, ...].shape 
            )
        display[t, rr, cc] = 1
        
# -----------------------------------------------------------------------------

# Display
viewer = napari.Viewer()
# viewer.add_image(pMaskAll, blending="additive", colormap="magenta", opacity=0.33) 
viewer.add_image(C1_min , blending="additive", opacity=0.66) 
viewer.add_image(maskAll, blending="additive", colormap="yellow", opacity=0.1, visible=False)
viewer.add_image(maskOut, blending="additive", colormap="bop blue", opacity=0.33, visible=False)
viewer.add_image(maskBod, blending="additive", colormap="bop orange", opacity=0.33, visible=False)
viewer.add_image(display, blending="additive")
viewer.add_labels(labels)

#%%



# -----------------------------------------------------------------------------

        
# viewer = napari.Viewer()
# viewer.add_image(C1_min, opacity=0.33)
# viewer.add_image(display, blending="additive")
# viewer.add_image(outl, blending="additive")
# viewer.add_labels(labels)
