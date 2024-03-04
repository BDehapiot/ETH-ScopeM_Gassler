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
exp_name, exp_numb = "R02", "002"

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

# Display
viewer = napari.Viewer()
viewer.add_image(C1_min,  blending="additive", opacity=0.33) 
viewer.add_image(predBod, blending="additive", colormap="bop blue") 
viewer.add_image(predOut, blending="additive", colormap="bop orange") 

#%%

# tmp1 = np.gradient(np.std(predBod, axis=(1, 2)))
# tmp2 = np.gradient(np.std(predOut, axis=(1, 2)))
# plt.plot(tmp1)
# plt.plot(tmp2)
# plt.plot((tmp1 + tmp2) / 2)

# tmp1 = np.gradient(np.std(C1_min, axis=(1, 2)))
# plt.plot(tmp1)

#%%

# from skimage.filters import gaussian
# from skimage.measure import label, regionprops

# # -----------------------------------------------------------------------------

# sigma = 0.5

# # -----------------------------------------------------------------------------

# # Get mask
# predict = (
#     gaussian(predBod, sigma=(0, sigma, sigma)) -
#     gaussian(predOut, sigma=(0, sigma, sigma)) * 0.5 # Parameters
#     )
# predict[predict < 0] = 0
# mask = predict > 0.25 # Parameters

# # Get labels
# labels = []
# labels.append(label(mask[0, ...]))
# for t in range(1, mask.shape[0]):
#     templabels = label(mask[t, ...])
#     props = regionprops(templabels)
#     for prop in props:
#         idx = (prop.coords[:, 0], prop.coords[:, 1])
#         values = labels[t - 1][idx]
#         values = values[values != 0]
#         values, counts = np.unique(values, return_counts=True)
#         if values.size == 0:
#             mode = 0
#         else:
#             mode = values[np.argmax(counts)]
#         templabels[idx] = mode
#     labels.append(templabels)   
# labels = np.stack(labels)
        
# # -----------------------------------------------------------------------------

# # Display
# viewer = napari.Viewer()
# viewer.add_image(C1_min)
# viewer.add_labels(labels)
# # viewer.add_image(predBod) 
# # viewer.add_image(predOut) 
# # viewer.add_image(predict) 
