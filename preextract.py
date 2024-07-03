#%% Imports -------------------------------------------------------------------

import gc
import nd2
import time
import numpy as np
from skimage import io
from pathlib import Path
from skimage.transform import downscale_local_mean

#%% Comments ------------------------------------------------------------------

# Stack format is TZCYX

#%% Inputs --------------------------------------------------------------------

# Paths
remote_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Gassler")
data_path = Path(remote_path, "data")

# Parameter(s)
exclude = [
    "20240327_tg_INDxRpxMr-MinimalMedium",
    "stock",
    ]
target_pixel_size = 0.433333333333332 

#%% Initialize ----------------------------------------------------------------

metadata = {
    "path" : [], "cond" : [], "date" : [], "oprt" : [], "numb" : []
    }

for path in list(remote_path.glob("**/*.nd2")):
    if not any(substring in str(path) for substring in exclude):
        cond, date, oprt, numb = path.stem.split("_")
        metadata["path"].append(path)
        metadata["cond"].append(cond)
        metadata["date"].append(date)
        metadata["oprt"].append(oprt)
        metadata["numb"].append(numb)

#%% Extract stacks ------------------------------------------------------------

for i, path in enumerate(metadata["path"]):
    
    # Check paths
    if "a.nd2" in path.name:
        path_b = path.with_name(path.name.replace("a.nd2", "b.nd2"))
        C1_path = Path(data_path, path.name.replace("a.nd2", "_C1.tif"))
        C2_path = Path(data_path, path.name.replace("a.nd2", "_C2.tif"))
    else:
        path_b = None
        C1_path = Path(data_path, path.name.replace(".nd2", "_C1.tif"))
        C2_path = Path(data_path, path.name.replace(".nd2", "_C2.tif"))
    
    if not C1_path.exists():
        
        if "b.nd2" not in path.name:
        
            # Open ------------------------------------------------------------    
        
            print(path.name)
            print("Open :", end='')
            t0 = time.time()
                  
            with nd2.ND2File(path) as ndfile:
                
                # Read metadata
                nT, nZ, nC, nY, nX = ndfile.shape
                pixel_size = ndfile.voxel_size()[0]
                downscale_factor = int(target_pixel_size // pixel_size)
                
                # Open data
                tmp = ndfile.asarray()
                C1, C2 = tmp[:,:,0,:,:], tmp[:,:,1,:,:]
                
            if path_b is not None:
                                
                with nd2.ND2File(path_b) as ndfile:
                    
                    # Open data
                    tmp_b = ndfile.asarray()
                    C1_b, C2_b = tmp_b[:,:,0,:,:], tmp_b[:,:,1,:,:]
                
                # Concatenate data
                C1 = np.concatenate((C1, C1_b), axis=0)
                C2 = np.concatenate((C2, C2_b), axis=0)
                
                del tmp_b
            
            del tmp
            
            t1 = time.time()
            print(f" {(t1-t0):<5.2f}s")
            
            # Format ----------------------------------------------------------

            print("Format :", end='')
            t0 = time.time()

            # Downscale data
            if downscale_factor > 1:
                C1 = downscale_local_mean(
                    C1, (1, 1, downscale_factor, downscale_factor))
                C2 = downscale_local_mean(
                    C2, (1, 1, downscale_factor, downscale_factor))
                
            # Normalize data
            # C1 = (C1 // 16).astype("uint8") # since max. int. = 4096
            # C2 = (C2 // 16).astype("uint8") # since max. int. = 4096

            t1 = time.time()
            print(f" {(t1-t0):<5.2f}s")
            
            # Save ------------------------------------------------------------
    
            print("Save :", end='')
            t0 = time.time()
        
            io.imsave(
                C1_path, C1.astype("uint16"), check_contrast=False,
                imagej=True,
                metadata={'axes': 'TZYX'},
                photometric='minisblack',
                planarconfig='contig',
                )
        
            io.imsave(
                C2_path, C2.astype("uint16"), check_contrast=False,
                imagej=True,
                metadata={'axes': 'TZYX'},
                photometric='minisblack',
                planarconfig='contig',
                )
            
            del C1, C2
            gc.collect()  
        
            t1 = time.time()
            print(f" {(t1-t0):<5.2f}s")