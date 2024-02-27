#%% Imports -------------------------------------------------------------------

import nd2
import time
import numpy as np
from skimage import io
from pathlib import Path
from skimage.transform import downscale_local_mean

#%% Comments ------------------------------------------------------------------

'''
- format TZCYX

'''

#%% Inputs --------------------------------------------------------------------

# Path
loc_path   = Path("D:/local_Gassler/data")
train_path = Path(Path.cwd(), 'data', 'train')
C1_paths = list(loc_path.glob("**/*C1*.tif"))

# Frame selection
np.random.seed(42)
nFrame = 10 # number of randomly selected frames

# Parameters
downscale_factor = 4

#%% Extract -------------------------------------------------------------------

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
    
    # Select random frames
    rIdxs = np.random.randint(0, C1.shape[0] // 2, size=nFrame)
    for idx in rIdxs:        
        io.imsave(
            Path(train_path, C1_path.name.replace(".tif", f"_{idx:03d}.tif")),
            C1_min[idx,...], check_contrast=False,
            )