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
exp_name, exp_numb = "IND", "004"

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
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, skeletonize, remove_small_objects

# -----------------------------------------------------------------------------

# Parameters
sigma = 1
threshAll = 0.05
threshOut = 0.25
threshBod = 0.05
min_size = 32

# -----------------------------------------------------------------------------

# Get masks
maskAll, maskOut, maskBod, outlOut = [], [], [], []
for t in range(C1_min.shape[0]):
    
    # Create masks
    mAll = gaussian(predAll[t, ...], sigma=sigma) > threshAll
    mOut = gaussian(predOut[t, ...], sigma=sigma) > threshOut
    mBod = gaussian(predBod[t, ...], sigma=sigma) > threshBod
    mAll = remove_small_objects(mAll, min_size=min_size)
    mOut = remove_small_objects(mOut, min_size=min_size)
    oOut = skeletonize(mOut, method="lee")
    # mAll[oOut == 255] = 0
    # mAll = remove_small_objects(mAll, min_size=min_size)
    
    # # Filter masks
    # for prop in regionprops(label(mAll, connectivity=1)):
    #     idx = (prop.coords[:, 0], prop.coords[:, 1])
    #     roundness = 4 * np.pi * prop.area / (prop.perimeter ** 2)
    #     if roundness < 0.5:
    #         mAll[idx] = 0
        
    # Append
    maskAll.append(mAll)
    maskOut.append(mOut)
    maskBod.append(mBod)
    outlOut.append(oOut)
    
maskAll = np.stack(maskAll)
maskOut = np.stack(maskOut)
maskBod = np.stack(maskBod)
outlOut = np.stack(outlOut)

# -----------------------------------------------------------------------------

# Display
viewer = napari.Viewer()
viewer.add_image(C1_min,  blending="additive", opacity=0.33) 
viewer.add_image(maskAll, blending="additive", colormap="magenta") 
viewer.add_image(maskOut, blending="additive", colormap="bop blue") 
viewer.add_image(maskBod, blending="additive", colormap="bop orange") 

#%%

# from skimage.filters import gaussian
# from skimage.measure import label, regionprops
# from skimage.morphology import binary_dilation, skeletonize, remove_small_objects

# -----------------------------------------------------------------------------

# # Parameters
# sigma = 1
# threshAll = 0.05
# threshOut = 0.25
# min_size = 32

# -----------------------------------------------------------------------------

# # Get masks
# maskAll, maskOut, outlOut, rndsMap = [], [], [], []
# for t in range(C1_min.shape[0]):
    
#     # Create masks
#     mAll = gaussian(predAll[t, ...], sigma=1) > threshAll
#     mOut = gaussian(predOut[t, ...], sigma=1) > threshOut
#     mAll = remove_small_objects(mAll, min_size=min_size)
#     mOut = remove_small_objects(mOut, min_size=min_size)
#     oOut = skeletonize(mOut, method="lee")
#     mAll[oOut == 255] = 0
#     mAll = remove_small_objects(mAll, min_size=min_size)
    
#     # Filter masks
#     for prop in regionprops(label(mAll, connectivity=1)):
#         idx = (prop.coords[:, 0], prop.coords[:, 1])
#         roundness = 4 * np.pi * prop.area / (prop.perimeter ** 2)
#         if roundness < 0.5:
#             mAll[idx] = 0
        
#     # Append
#     maskAll.append(mAll)
#     maskOut.append(mOut)
#     outlOut.append(oOut)
    
# maskAll = np.stack(maskAll)
# maskOut = np.stack(maskOut)
# outlOut = np.stack(outlOut)

# # Get labels
# labels = []
# labels.append(label(maskAll[0, ...], connectivity=1))
# for t in range(1, maskAll.shape[0]):
#     labs = label(maskAll[t, ...], connectivity=1)
#     med = np.median(np.stack(labels), axis=0)
#     props = regionprops(labs)    
#     for prop in props:
#         idx = (prop.coords[:, 0], prop.coords[:, 1])
#         values = med[idx]
#         values = values[values != 0]
#         values, counts = np.unique(values, return_counts=True)
#         mode = 0 if values.size == 0 else values[np.argmax(counts)]
#         labs[idx] = mode
#     labels.append(labs)
# labels = np.stack(labels)

# Display
# viewer = napari.Viewer()
# viewer.add_image(C1_min,  blending="additive", opacity=0.33) 
# viewer.add_image(maskAll, blending="additive", colormap="bop blue", opacity=0.5)
# viewer.add_image(maskOut, blending="additive", colormap="bop orange", opacity=0.5)
# viewer.add_image(outlOut, blending="additive", colormap="bop orange")
# viewer.add_labels(labels)

#%%

# objectData = []
# for mAll in maskAll:
#     lab = label(mAll, connectivity=1)
#     objectData.append(len(np.unique(lab)))
# nObject = np.stack(objectData)
# tf = np.argmax(np.gradient(nObject))
# # plt.plot(np.gradient(nObject))


