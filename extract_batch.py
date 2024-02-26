#%% Imports -------------------------------------------------------------------

import gc
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
nd2_paths[0:2] = [] # Remove merged .nd2 files

# Stems
stems = []
for path in nd2_paths:
    stems.append(path.stem)
stems = sorted(set(stems))

# Parameters
downscale_factor = 4

#%% Extract -------------------------------------------------------------------

for stem in stems:
    
    exp_name, exp_numb = stem.split("_")
    paths = list(net_path.glob(f"**/{exp_name}_{exp_numb}.nd2"))
    C1_path = Path(loc_path, f"C1_{exp_name}_{exp_numb}.tif")
    C2_path = Path(loc_path, f"C2_{exp_name}_{exp_numb}.tif")
    
    if not C1_path.exists(): 
    
        C1, C2 = [], []
        for path in paths:

            with nd2.ND2File(path) as ndfile:
        
                print(f"Open - {path.parent.name}/{path.name} :", end='')
                t0 = time.time()
                
                tmp = ndfile.asarray()
                tmpC1, tmpC2 = tmp[:,:,0,:,:], tmp[:,:,1,:,:]
                tmpC1 = downscale_local_mean(
                    tmpC1, (1, 1, downscale_factor, downscale_factor))
                tmpC2 = downscale_local_mean(
                    tmpC2, (1, 1, downscale_factor, downscale_factor))
                tmpC1 = (tmpC1 // 16).astype("uint8") # since max. int. = 4096
                tmpC2 = (tmpC2 // 16).astype("uint8") # since max. int. = 4096
                C1.append(tmpC1); C2.append(tmpC2)
                
                del tmp, tmpC1, tmpC2
                
                t1 = time.time()
                print(f" {(t1-t0):<5.2f}s")     
                
        C1 = np.concatenate(C1, axis=0)
        C2 = np.concatenate(C2, axis=0)
    
        print("Save :", end='')
        t0 = time.time()
    
        io.imsave(
            C1_path, C1, check_contrast=False,
            imagej=True,
            metadata={'axes': 'TZYX'},
            photometric='minisblack',
            planarconfig='contig',
            )
    
        io.imsave(
            C2_path, C2, check_contrast=False,
            imagej=True,
            metadata={'axes': 'TZYX'},
            photometric='minisblack',
            planarconfig='contig',
            )
    
        t1 = time.time()
        print(f" {(t1-t0):<5.2f}s")
        
        del C1, C2
        gc.collect()                