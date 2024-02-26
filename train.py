#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

#%% Inputs --------------------------------------------------------------------

# Path
loc_path = Path("D:/local_Gassler/data")
# C1_paths = list(loc_path.glob("*C1*.tif"))
# C2_paths = list(loc_path.glob("*C2*.tif"))

C1_paths = list(loc_path.glob("*C1_R02_005*.tif"))

#%%

for C1_path in C1_paths:
    
    C1 = io.imread(C1_path).astype("float32")
    
    # Mean normalization
    for z in range(C1.shape[1]):
        for t in range(C1.shape[0]):
            C1[t,z,...] /= np.mean(C1[t,z,...])
            
    # Min. projection & 0 to 1 normalization
    C1_min = np.min(C1, axis=1)        
    pMax = np.percentile(C1_min, 99.9)
    C1_min[C1_min > pMax] = pMax
    C1_min = (C1_min / pMax).astype(float)
            
#%%

import napari
viewer = napari.Viewer()
viewer.add_image(C1_min)
            
    