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
model_bodies_path = Path(Path.cwd(), f"model_weights_bodies.h5")
model_outlines_path = Path(Path.cwd(), f"model_weights_outlines.h5")
exp_name, exp_numb = "IND", "003"

# Patches
downscale_factor = 4
size = 512 // downscale_factor
overlap = size // 4

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
predict_bodies = model.predict(patches).squeeze()
model.load_weights(model_outlines_path) 
predict_outlines = model.predict(patches).squeeze()

# Merge patches
print("Merge patches   :", end='')
t0 = time.time()
predict_bodies = merge_patches(predict_bodies, C1_min.shape, size, overlap)
predict_outlines = merge_patches(predict_outlines, C1_min.shape, size, overlap)
t1 = time.time()
print(f" {(t1-t0):<5.2f}s") 

# Display
viewer = napari.Viewer()
viewer.add_image(predict_bodies) 
viewer.add_image(predict_outlines) 

#%%

# from skimage.measure import label
# from skimage.segmentation import watershed

# # -----------------------------------------------------------------------------

# markers = label(predict > 0.5) + 1
# borders = (predict > 0.001) & (predict < 0.5)
# markers[borders == True] = 0

# wat = []
# for t in range(markers.shape[0]):
#     tmp = watershed(
#         predict[t,...], 
#         markers=markers[t,...],
#         compactness=10,
#         watershed_line=True,
#         )
#     wat.append(tmp)
# wat = np.stack(wat)
# # wat[mask == 0] = 0

# # Display
# viewer = napari.Viewer()
# viewer.add_image(predict) 
# viewer.add_labels(markers) 
# viewer.add_labels(wat)

# # Display
# viewer = napari.Viewer()
# viewer.add_image(predict) 
# viewer.add_image(markers, colormap="green", opacity=0.33) 
# viewer.add_image(borders, colormap="magenta", opacity=0.33) 
