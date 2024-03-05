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
from skimage.morphology import binary_dilation, skeletonize, remove_small_objects

# -----------------------------------------------------------------------------

# Do the same but processing in 2D (problem with removing small object for expl)

maskBod = gaussian(predBod, sigma=(0, 1, 1)) > 0.1
maskOut = gaussian(predOut, sigma=(0, 1, 1)) > 0.25
maskBod = remove_small_objects(maskBod, min_size=128)
maskOut = remove_small_objects(maskOut, min_size=128)

outlOut = []
for t in range(maskOut.shape[0]):
    outlOut.append(skeletonize(maskOut[t, ...], method="lee"))
outlOut = np.stack(outlOut)
maskBod[outlOut == 255] = 0
maskBod = remove_small_objects(maskBod, min_size=512, connectivity=0)

# Display
viewer = napari.Viewer()
viewer.add_image(maskBod, blending="additive", colormap="bop blue", opacity=0.5)
viewer.add_image(maskOut, blending="additive", colormap="bop orange", opacity=0.5)
viewer.add_image(outlOut, blending="additive", colormap="bop orange")

# -----------------------------------------------------------------------------

# objectData = []
# for msk in mask:
#     lab = label(msk)
#     objectData.append(len(np.unique(lab)))
# nObject = np.stack(objectData)
# plt.plot(np.gradient(nObject))