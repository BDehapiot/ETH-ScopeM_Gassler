#%% Imports -------------------------------------------------------------------

import time
import pickle
import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt

# Functions
from functions import preprocess, predict, process
    
#%% Inputs --------------------------------------------------------------------

# Paths
exp_name = "IND_003"
measures = ["area", "length", "roundness", "intensity"]
local_path = Path("D:/local_Gassler/data")
save_path = Path(local_path, exp_name)
save_path.mkdir(exist_ok=True)

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

print("hello")

#%% Preprocessing -------------------------------------------------------------

C1 = io.imread(Path(local_path, f"{exp_name}_C1.tif"))
C2 = io.imread(Path(local_path, f"{exp_name}_C2.tif"))
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
mask, labels, skel, display, data = process(
    C1_proj, C2_proj,
    predAll, predOut, predBod,
    threshAll, threshOut, threshBod,
    min_size, min_roundness
    )
t1 = time.time()
print(f"Process : {(t1-t0):<.2f}s")

#%% Save ----------------------------------------------------------------------

# Images
# io.imsave(
#     Path(save_path, f"{exp_name}_labels.tif"),
#     labels.astype("uint8"), check_contrast=False,
#     )
io.imsave(
    Path(save_path, f"{exp_name}_labels.tif"),
    labels.astype("uint8"), check_contrast=False,
    )
io.imsave(
    Path(save_path, f"{exp_name}_display.tif"),
    display.astype("uint8"), check_contrast=False,
    )

# Data as PKL
with open(Path(save_path, f"{exp_name}_data.pkl"), 'wb') as file:
    pickle.dump(data, file)

# Data as CSV
for measure in measures:
    np.savetxt(
        Path(save_path, f"{exp_name}_{measure}.csv"),
        np.stack([d[f"{measure}"] for d in data], axis=1), 
        delimiter=",", fmt="%.3f",
        )

#%% Plot ----------------------------------------------------------------------

plt.figure(figsize=(6, 12))

for i, measure in enumerate(measures):
    plt.subplot(len(measures), 1, i + 1)
    for d in data:
        plt.plot(d[f"{measure}"])
    plt.title(f"{measure}")
    plt.xlabel("index")
    plt.ylabel(f"{measure}")

plt.tight_layout()
plt.show()

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_image(C1_proj, blending="additive", opacity=0.5)
viewer.add_image(C2_proj, blending="additive", colormap="yellow")
viewer.add_image(display, blending="additive")
viewer.add_labels(labels, visible=False)

