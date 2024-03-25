#%% Imports -------------------------------------------------------------------

import numpy as np
from skimage import io
from pathlib import Path

# Functions
# from functions import preprocess

#%% Comments ------------------------------------------------------------------

'''
- format TZCYX

'''

#%% Inputs --------------------------------------------------------------------

# Path
loc_path = Path("D:/local_Gassler/data")
train_path = Path(Path.cwd(), 'data', 'train')
C1_paths = list(loc_path.glob("**/*C1*.tif"))

# Frame selection
np.random.seed(42)
nFrame = 60 # number of randomly selected frames

# Parameters
downscale_factor = 4

#%% Extract -------------------------------------------------------------------

def preprocess(hstack):
    
    # Convert to float32
    hstack = hstack.astype("float32")
    
    # Mean normalization
    for z in range(hstack.shape[1]):
        for t in range(hstack.shape[0]):
            hstack[t,z,...] /= np.mean(hstack[t,z,...])
            
    # Min. projection & 0 to 1 normalization
    hstack_min = np.std(hstack, axis=1)        
    pMax = np.percentile(hstack_min, 99.9)
    hstack_min[hstack_min > pMax] = pMax
    hstack_min = (hstack_min / pMax).astype("float32")
    
    return hstack_min

# Open and preprocess
C1 = io.imread(C1_paths[0])
C1_min = preprocess(C1)

# Display
import napari
viewer = napari.Viewer()
viewer.add_image(C1_min)

