#%% Imports -------------------------------------------------------------------

import time
import napari
import numpy as np
from skimage import io
from pathlib import Path
import segmentation_models as sm

# Functions
from functions import preprocess, get_patches, merge_patches

#%% Inputs --------------------------------------------------------------------

# Paths
loc_path = Path("D:/local_Gassler/data")
model_path = Path(Path.cwd(), "model_weights.h5")
exp_name, exp_numb = "IND", "001"

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

# Display
viewer = napari.Viewer()
viewer.add_image(np.stack(C1_min)) 
viewer.add_image(np.stack(predict))

test = patches[0] 
