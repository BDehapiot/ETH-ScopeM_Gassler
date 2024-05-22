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

# Paths
remote_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Gassler")
data_path = Path(remote_path, "data")

# Parameter
target_pixel_size = 0.433333333333332 

#%% Initialize ----------------------------------------------------------------

metadata = {
    "path" : [], 
    "cond" : [], 
    "date" : [], 
    "oprt" : [], 
    "numb" : [],
    }

for path in list(remote_path.glob("**/*.nd2")):
    cond, date, oprt, numb = path.stem.split("_")
    metadata["path"].append(path)
    metadata["cond"].append(cond)
    metadata["date"].append(date)
    metadata["oprt"].append(oprt)
    metadata["numb"].append(numb)

#%% Extract stacks ------------------------------------------------------------

for i, path in enumerate(metadata["path"]):
    
    # Check name
    name = path.name
    if "a.nd2" in name:
        path_b = path.with_name(name.replace("a.nd2", "b.nd2"))
        name = name.replace("a.nd2", ".tif")
    C1_path = Path(data_path, name.replace(".tif", "_C1.tif"))
    C2_path = Path(data_path, name.replace(".tif", "_C2.tif"))
    
    if not C1_path.exists():
        
        if path.stem == "pdINDw_20240403_tg_001":
        
            # Open ------------------------------------------------------------    
        
            # print(path.name)
            # print("Open :", end='')
            t0 = time.time()
                  
            with nd2.ND2File(path) as ndfile:
                
                # Read metadata
                nT, nZ, nC, nY, nX = ndfile.shape
                pixel_size = ndfile.voxel_size()[0]
                downscale_factor = int(target_pixel_size // pixel_size)
                
                # # Open data
                # tmp = ndfile.asarray()
                # tmpC1, tmpC2 = tmp[:,:,0,:,:], tmp[:,:,1,:,:]
                
                # # Downscale data
                # if downscale_factor > 1:
                #     tmpC1 = downscale_local_mean(
                #         tmpC1, (1, 1, downscale_factor, downscale_factor))
                #     tmpC2 = downscale_local_mean(
                #         tmpC2, (1, 1, downscale_factor, downscale_factor))
                    
                # # Normalize data
                # tmpC1 = (tmpC1 // 16).astype("uint8") # since max. int. = 4096
                # tmpC2 = (tmpC2 // 16).astype("uint8") # since max. int. = 4096
            
            # t1 = time.time()
            # print(f" {(t1-t0):<5.2f}s")
    
            # Save ------------------------------------------------------------
    
            # print("Save :", end='')
            # t0 = time.time()
        
            # io.imsave(
            #     C1_path, tmpC1, check_contrast=False,
            #     imagej=True,
            #     metadata={'axes': 'TZYX'},
            #     photometric='minisblack',
            #     planarconfig='contig',
            #     )
        
            # io.imsave(
            #     C2_path, tmpC2, check_contrast=False,
            #     imagej=True,
            #     metadata={'axes': 'TZYX'},
            #     photometric='minisblack',
            #     planarconfig='contig',
            #     )
        
            # t1 = time.time()
            # print(f" {(t1-t0):<5.2f}s")

#%%


# test = ndfile.metadata

# nT, nZ, nC, nY, nX = tmp.shape
# tmpC1, tmpC2 = tmp[:,:,0,:,:], tmp[:,:,1,:,:]


#%% Extract stacks ------------------------------------------------------------

# for stem in stems:
    
#     exp_name, exp_numb = stem.split("_")
#     paths = list(remote_path.glob(f"**/{exp_name}_{exp_numb}.nd2"))
#     C1_path = Path(local_path, f"{exp_name}_{exp_numb}_C1.tif")
#     C2_path = Path(local_path, f"{exp_name}_{exp_numb}_C2.tif")
    
#     if not C1_path.exists():
    
#         C1, C2 = [], []
#         for path in paths:

#             with nd2.ND2File(path) as ndfile:
        
#                 print(f"Open - {path.parent.name}/{path.name} :", end='')
#                 t0 = time.time()
                
#                 tmp = ndfile.asarray()
#                 tmpC1, tmpC2 = tmp[:,:,0,:,:], tmp[:,:,1,:,:]
#                 tmpC1 = downscale_local_mean(
#                     tmpC1, (1, 1, downscale_factor, downscale_factor))
#                 tmpC2 = downscale_local_mean(
#                     tmpC2, (1, 1, downscale_factor, downscale_factor))
#                 tmpC1 = (tmpC1 // 16).astype("uint8") # since max. int. = 4096
#                 tmpC2 = (tmpC2 // 16).astype("uint8") # since max. int. = 4096
#                 C1.append(tmpC1); C2.append(tmpC2)
                
#                 del tmp, tmpC1, tmpC2
                
#                 t1 = time.time()
#                 print(f" {(t1-t0):<5.2f}s")     
                
#         C1 = np.concatenate(C1, axis=0)
#         C2 = np.concatenate(C2, axis=0)
    
#         print("Save :", end='')
#         t0 = time.time()
    
#         io.imsave(
#             C1_path, C1, check_contrast=False,
#             imagej=True,
#             metadata={'axes': 'TZYX'},
#             photometric='minisblack',
#             planarconfig='contig',
#             )
    
#         io.imsave(
#             C2_path, C2, check_contrast=False,
#             imagej=True,
#             metadata={'axes': 'TZYX'},
#             photometric='minisblack',
#             planarconfig='contig',
#             )
    
#         t1 = time.time()
#         print(f" {(t1-t0):<5.2f}s")
        
#         del C1, C2
#         gc.collect()                