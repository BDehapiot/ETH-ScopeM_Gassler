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
model_bodies_path = Path(Path.cwd(), "model_weights_bodies.h5")
model_outlines_path = Path(Path.cwd(), "model_weights_outlines.h5")
exp_name, exp_numb = "R20", "001"

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
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Load weights & predict
model.load_weights(model_bodies_path) 
predBod = model.predict(patches).squeeze()
model.load_weights(model_outlines_path) 
predOut = model.predict(patches).squeeze()

# Merge patches
print("Merge patches   :", end='')
t0 = time.time()
predBod = merge_patches(predBod, C1_min.shape, size, overlap)
predOut = merge_patches(predOut, C1_min.shape, size, overlap)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# # Display
# viewer = napari.Viewer()
# viewer.add_image(C1_min,  blending="additive", opacity=0.33) 
# viewer.add_image(predBod, blending="additive", colormap="bop blue") 
# viewer.add_image(predOut, blending="additive", colormap="bop orange") 

#%%

from skimage.filters import gaussian
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation

# -----------------------------------------------------------------------------

sigma = (0, 1, 1)

# -----------------------------------------------------------------------------

# Get mask 
mask = predBod > 0.1
mask[predOut > 0.25] = 0
mask = gaussian(mask, sigma=sigma) > 0.5

# Get outlines
outl = []
for msk in mask:
    outl.append(binary_dilation(msk) ^ msk)
outl = np.stack(outl)

# Get labels
labels = []
labels.append(label(mask[0, ...]))
for t in range(1, mask.shape[0]):
    labs = label(mask[t, ...])
    med = np.median(np.stack(labels), axis=0)
    props = regionprops(labs)    
    for prop in props:
        idx = (prop.coords[:, 0], prop.coords[:, 1])
        values = med[idx]
        values = values[values != 0]
        values, counts = np.unique(values, return_counts=True)
        mode = 0 if values.size == 0 else values[np.argmax(counts)]
        labs[idx] = mode
    labels.append(labs)
labels = np.stack(labels)

# -----------------------------------------------------------------------------

# Display
viewer = napari.Viewer()
viewer.add_image(C1_min, opacity=0.33)
viewer.add_image(predBod, colormap="bop blue", blending="additive")
viewer.add_image(predOut, colormap="bop orange", blending="additive")
viewer.add_image(outl, blending="additive")
# viewer.add_labels(labels)

#%%

from skimage.draw import rectangle_perimeter

# -----------------------------------------------------------------------------

display = np.zeros_like(labels)
for t in range(labels.shape[0]):
    # labs = labels[t, ...]
    labs = label(mask[t, ...])
    for lab in np.unique(labs):
        idx = np.where(labs == lab)
        start = (np.min(idx[0]), np.min(idx[1]))
        end = (np.max(idx[0]), np.max(idx[1]))
        rr, cc = rectangle_perimeter(start, end,
            shape=display[t, ...].shape 
            )
        display[t, rr, cc] = 1
        
viewer = napari.Viewer()
viewer.add_image(C1_min, opacity=0.33)
viewer.add_image(display, blending="additive")
viewer.add_image(outl, blending="additive")
viewer.add_labels(labels)