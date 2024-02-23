#%% Imports -------------------------------------------------------------------

import nd2
import time
import numpy as np
from skimage import io
from pathlib import Path
from skimage.transform import downscale_local_mean

#%%

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
exp_name, exp_numb = "IND", "002" 

# Parameters
downscale_factor = 4


#%% Extract -------------------------------------------------------------------

hstack = []
for path in nd2_paths:
    if (exp_name in path.name) and (exp_numb in path.name):
        with nd2.ND2File(path) as ndfile:

            print(f"Open - {path.parent.name}/{path.name} :", end='')
            t0 = time.time()
            
            tmp = ndfile.asarray()[:,:,0,:,:]
            tmp = downscale_local_mean(tmp, (1, 1, downscale_factor, downscale_factor))
            tmp = (tmp // 16).astype("uint8") 
            hstack.append(tmp)
            
            t1 = time.time()
            print(f" {(t1-t0):<5.2f}s")
            
hstack = np.concatenate(hstack, axis=0)
            
print("Save :", end='')
t0 = time.time()

io.imsave(
    Path(loc_path, f"{exp_name}_{exp_numb}_.tif"),
    hstack, 
    check_contrast=False,
    imagej=True,
    metadata={'axes': 'TZYX'},
    photometric='minisblack',
    planarconfig='contig',
    )

t1 = time.time()
print(f" {(t1-t0):<5.2f}s")
