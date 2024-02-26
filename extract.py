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
loc_path  = Path("D:/local_Gassler/data")
net_path  = Path(r"\\scopem-userdata.ethz.ch\Image-Clinic\20240131_Burst_Rate_24TG02")
nd2_paths = list(net_path.glob("**/*.nd2"))

# Select data
# IND (001 to 005), R02 (001 to 006), R11 (001 to 005), R20 (001 to 005)
exp_name, exp_numb = "R11", "001" 

# Parameters
downscale_factor = 4

#%% Extract -------------------------------------------------------------------

C1, C2 = [], []
for path in nd2_paths:
    if (exp_name in path.name) and (exp_numb in path.name):
        with nd2.ND2File(path) as ndfile:

            print(f"Open - {path.parent.name}/{path.name} :", end='')
            t0 = time.time()
            
            tmp = ndfile.asarray()
            tmpC1, tmpC2 = tmp[:,:,0,:,:], tmp[:,:,1,:,:]
            tmpC1 = downscale_local_mean(tmpC1, (1, 1, downscale_factor, downscale_factor))
            tmpC2 = downscale_local_mean(tmpC2, (1, 1, downscale_factor, downscale_factor))
            tmpC1 = (tmpC1 // 16).astype("uint8") # since max. int. = 4096
            tmpC2 = (tmpC2 // 16).astype("uint8") # since max. int. = 4096
            C1.append(tmpC1); C2.append(tmpC2)
            
            t1 = time.time()
            print(f" {(t1-t0):<5.2f}s")
            
C1 = np.concatenate(C1, axis=0)
C2 = np.concatenate(C2, axis=0)

#%% Save ----------------------------------------------------------------------
    
print("Save :", end='')
t0 = time.time()

io.imsave(
    Path(loc_path, f"C1_{exp_name}_{exp_numb}.tif"),
    C1, check_contrast=False,
    imagej=True,
    metadata={'axes': 'TZYX'},
    photometric='minisblack',
    planarconfig='contig',
    )

io.imsave(
    Path(loc_path, f"C2_{exp_name}_{exp_numb}.tif"),
    C2, check_contrast=False,
    imagej=True,
    metadata={'axes': 'TZYX'},
    photometric='minisblack',
    planarconfig='contig',
    )

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")
