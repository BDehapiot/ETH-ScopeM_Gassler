#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import preprocess, predict, process
    
#%% Inputs --------------------------------------------------------------------

# Paths
local_path = Path("D:/local_Gassler/data")
exp_name, exp_numb = "IND", "002"

# Patches
downscale_factor = 4
size = 512 // downscale_factor
overlap = size // 2

# Parameters
threshAll = 0.05 #
threshOut = 0.25 #
threshBod = 0.05 #
min_size = 32 # 
min_roundness = 0.3 #

#%% Preprocessing -------------------------------------------------------------

C1 = io.imread(Path(local_path, f"{exp_name}_{exp_numb}_C1.tif"))
C2 = io.imread(Path(local_path, f"{exp_name}_{exp_numb}_C2.tif"))
C1_proj = preprocess(C1)
C2_proj = np.sum(C2, axis=1)

#%% Execute -------------------------------------------------------------------

t0 = time.time()
predAll, predOut, predBod = predict(
    C1_proj, size, overlap
    )
t1 = time.time()
print(f"Predict : {(t1-t0):<.2f}s")

t0 = time.time()
mask, labels, display, data = process(
    C1_proj, C2_proj,
    predAll, predOut, predBod,
    threshAll, threshOut, threshBod,
    min_size, min_roundness
    )
t1 = time.time()
print(f"Process : {(t1-t0):<.2f}s")

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(C1_proj, blending="additive", opacity=0.5)
viewer.add_image(C2_proj, blending="additive", colormap="yellow")
viewer.add_image(display, blending="additive")
viewer.add_labels(labels, visible=False)

#%% Save ----------------------------------------------------------------------

# Images
io.imsave(
    Path(local_path, Path(local_path, f"{exp_name}_{exp_numb}_labels.tif")),
    labels.astype("uint8"), check_contrast=False,
    )
io.imsave(
    Path(local_path, Path(local_path, f"{exp_name}_{exp_numb}_display.tif")),
    display.astype("uint8"), check_contrast=False,
    )

# Data
with open(Path(local_path, f"{exp_name}_{exp_numb}_data.pkl"), 'wb') as file:
    pickle.dump(data, file)
    
#%% Plot ----------------------------------------------------------------------

import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------

plt.figure(figsize=(6, 12))

plt.subplot(4, 1, 1)
for d in data:
    plt.plot(d["area"])
plt.title('Areas')
plt.xlabel('Index')
plt.ylabel('Area')

plt.subplot(4, 1, 2)
for d in data:
    plt.plot(d["length"])
plt.title('length')
plt.xlabel('Index')
plt.ylabel('length')

plt.subplot(4, 1, 3)
for d in data:
    plt.plot(d["roundness"])
plt.title('Roundness')
plt.xlabel('Index')
plt.ylabel('Roundness')

plt.subplot(4, 1, 4)
for d in data:
    plt.plot(d["intensity"])
plt.title('Intensities')
plt.xlabel('Index')
plt.ylabel('Intensity')

plt.tight_layout()
plt.show()

#%% Tests ---------------------------------------------------------------------

#%%

# from skimage.measure import label, regionprops
# from bdtools.skel import pixconn

# # -----------------------------------------------------------------------------

# arr = np.array(
#     [[0, 1, 0, 0, 0, 0],
#      [0, 1, 0, 0, 1, 0],
#      [0, 1, 0, 1, 0, 0],
#      [0, 0, 1, 0, 0, 0],
#      [0, 0, 1, 1, 0, 0],
#      [0, 1, 0, 1, 0, 0]])

# length = 0
# lab = label(arr, connectivity=1)
# props = regionprops(lab)
# for prop in props:
#     length += prop.area - 1
# length += (np.max(lab) - 1) * np.sqrt(2)



