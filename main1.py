#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm
from joblib import Parallel, delayed 

# Functions
from functions import preprocess, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
loc_path = Path("D:/local_Gassler/data")
model_path = Path(Path.cwd(), "model_weights.h5")
exp_name, exp_numb = "IND", "003"

# Patches
downscale_factor = 4
size = 512 // downscale_factor
overlap = size // 8

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

# Load weights
model.load_weights(model_path) 

# Predict
predict = model.predict(patches).squeeze()

# Merge patches
print("Merge patches   :", end='')
t0 = time.time()
predict = merge_patches(predict, C1_min.shape, size, overlap)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

#%%

from skimage.measure import label
from skimage.filters import gaussian
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.morphology import square, binary_dilation

# -----------------------------------------------------------------------------

# Parameters
sigma = 1
thresh = 0.1
mdist = 5
prom = 0.5

# -----------------------------------------------------------------------------

def get_objects(predict, sigma, thresh, mdist, prom):
    
    # Nested functions --------------------------------------------------------
    
    def _get_locmax(img):
        coords = peak_local_max(img, min_distance=mdist, threshold_abs=prom)
        coords = tuple((coords[:, 0], coords[:, 1]))
        locmax = np.zeros_like(img)
        locmax[coords] = 1
        locmax = binary_dilation(locmax, footprint=square(5)) # to be removed
        return locmax
    
    # Execute -----------------------------------------------------------------
    
    predict = gaussian(predict, sigma=(0, sigma, sigma))
    mask = predict > thresh
    
    locmax = Parallel(n_jobs=-1)(
        delayed(_get_locmax)(img)
        for img in predict
        )
    locmax = np.stack(locmax)
    
    return mask, locmax
        
# -----------------------------------------------------------------------------

mask, locmax = get_objects(predict, sigma, thresh, mdist, prom)

# -----------------------------------------------------------------------------

wat = []
for t in range(mask.shape[0]):
    
    tmp = watershed(
        predict[t,...], 
        # markers=label(locmax[t,...]),
        compactness=10,
        watershed_line=True,
        )
    
    wat.append(tmp)
    
wat = np.stack(wat)
wat[mask == 0] = 0

# -----------------------------------------------------------------------------

# Display
viewer = napari.Viewer()
viewer.add_labels(wat) 
# viewer.add_image(locmax)

# # Display
# viewer = napari.Viewer()
# viewer.add_image(predict, opacity=0.5) 
# viewer.add_image(mask, blending="additive", colormap="red", opacity=0.5) 
# viewer.add_image(locmax, blending="additive", colormap="gray") 
