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
exp_name = "pdR10H_20240403_tg_002"
remote_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Gassler")
data_path = Path(remote_path, "data")
save_path = Path(data_path, exp_name)
save_path.mkdir(exist_ok=True)

# Patches
downscale_factor = 4
size = 512 // downscale_factor
overlap = size // 2

# Parameters
measures = ["area", "length", "roundness", "intensity"]
threshAll = 0.05 #
threshOut = 0.25 #
threshBod = 0.05 #
min_size = 32 # 
min_roundness = 0.3 #

#%% Preprocessing -------------------------------------------------------------

C1 = io.imread(Path(data_path, f"{exp_name}_C1.tif"))
C2 = io.imread(Path(data_path, f"{exp_name}_C2.tif"))
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

#%% Plot ----------------------------------------------------------------------

fig = plt.figure(figsize=(8, 16))
for m, measure in enumerate(measures):
    plt.subplot(len(measures), 1, m + 1)
    label = np.arange(1, len(measures) + 1)
    for d, dat in enumerate(data):
        plt.plot(dat[f"{measure}"], label=d + 1)
        if m == 0: fig.legend()
    plt.title(f"{measure}")
    plt.ylabel(f"{measure}")
    plt.xlabel("timepoint")      
plt.tight_layout()
plt.savefig(Path(save_path, f"{exp_name}_plot.jpg"), format='jpg')
plt.show()

#%% Save ----------------------------------------------------------------------

# Images
io.imsave(
    Path(save_path, f"{exp_name}_C1_proj.tif"),
    C1_proj.astype("float32"), check_contrast=False,
    )
io.imsave(
    Path(save_path, f"{exp_name}_C2_proj.tif"),
    C2_proj.astype("uint16"), check_contrast=False,
    )
io.imsave(
    Path(save_path, f"{exp_name}_labels.tif"),
    labels.astype("uint8"), check_contrast=False,
    )
io.imsave(
    Path(save_path, f"{exp_name}_display.tif"),
    display.astype("uint8"), check_contrast=False,
    )

# Composite
composite = np.concatenate((
    np.expand_dims(C1_proj / np.max(C1_proj) * 128, axis=1),
    np.expand_dims(C2_proj / np.max(C2_proj) * 255, axis=1),
    np.expand_dims(display, axis=1),
    ), axis=1)

val_range = np.arange(256, dtype='uint8')
lut_gray = np.stack([val_range, val_range, val_range])
lut_yellow = np.zeros((3, 256), dtype='uint8')
lut_yellow[[0, 1], :] = np.arange(256, dtype='uint8')

io.imsave(
    Path(save_path, f"{exp_name}_composite.tif"),
    composite.astype("uint8"),
    check_contrast=False,
    imagej=True,
    metadata={
        'axes': 'TCYX', 
        'mode': 'composite',
        'LUTs': [lut_gray, lut_yellow, lut_gray],
        }
    )

# Data as PKL
with open(Path(save_path, f"{exp_name}_data.pkl"), 'wb') as file:
    pickle.dump(data, file)

# Data as CSV
for measure in measures:
    headers = ','.join(f"{i+1}" for i in range(len(data)))
    np.savetxt(
        Path(save_path, f"{exp_name}_{measure}.csv"),
        np.stack([dat[f"{measure}"] for dat in data], axis=1), 
        delimiter=",", fmt="%.3f", header=headers, comments='',
        )

#%% Display -------------------------------------------------------------------

viewer = napari.Viewer()
viewer.add_labels(labels, visible=False)
viewer.add_image(predAll, blending="additive", colormap="magenta", visible=False)
viewer.add_image(predOut, blending="additive", colormap="magenta", visible=False)
viewer.add_image(predBod, blending="additive", colormap="magenta", visible=False)
viewer.add_image(C1_proj, blending="additive", opacity=0.5)
viewer.add_image(C2_proj, blending="additive", colormap="yellow")
viewer.add_image(display, blending="additive")