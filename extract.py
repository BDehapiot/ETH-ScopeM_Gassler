#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# Functions
from functions import preprocess

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
nFrame = 60 # number of randomly selected frames

# Parameters
downscale_factor = 4

#%% Extract -------------------------------------------------------------------

for C1_path in C1_paths:
    
    # Open and preprocess
    C1 = io.imread(C1_path)
    C1_proj = preprocess(C1)
        
    # Select random frames
    rIdxs = np.random.randint(0, C1.shape[0] // 2, size=nFrame)
    for idx in rIdxs:        
        io.imsave(
            Path(train_path, C1_path.name.replace(".tif", f"_{idx:03d}.tif")),
            C1_proj[idx,...], check_contrast=False,
            )