#%% Imports -------------------------------------------------------------------

import napari
import numpy as np
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from joblib import Parallel, delayed 

# Functions
from functions import get_patches

#%% Inputs --------------------------------------------------------------------

# Path
train_path = Path(Path.cwd(), 'data', 'train')

#%%

msks, imgs = [], []
for path in train_path.iterdir():
    if 'mask' in path.name:
        
        # Open data
        msk = io.imread(path)
        img = io.imread(str(path).replace('_mask', ''))
        
        # Append
        msks.append(msk)
        imgs.append(img)            
       
msks = np.stack(msks)
imgs = np.stack(imgs)
       
# # Display 
# viewer = napari.Viewer()
# viewer.add_image(msks)
# viewer.add_image(imgs) 
        
#%%

from skimage.filters import gaussian
from skimage.morphology import skeletonize, binary_erosion
from scipy.ndimage import distance_transform_edt

msk = msks[9]
labels = np.unique(msk)[1:]
borders = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
for l, label in enumerate(labels):
    tmp = msk == label
    tmp = tmp ^ binary_erosion(tmp)
    # tmp = gaussian(tmp.astype("float32"), sigma=2)
    borders[l,...] = tmp
borders = np.max(borders, axis=0)

# labels = np.unique(msk)[1:]
# edm = np.zeros((labels.shape[0], msk.shape[0], msk.shape[1]))
# for l, label in enumerate(labels):
#     tmp = msk == label
#     tmp = distance_transform_edt(tmp)
#     pMax = np.percentile(tmp[tmp > 0], 99.9)
#     tmp[tmp > pMax] = pMax
#     tmp = (tmp / pMax).astype("float32")
#     edm[l,...] = tmp
# edm = np.max(edm, axis=0)  
# return edm


    
# Display 
viewer = napari.Viewer()
viewer.add_labels(msk)
viewer.add_image(borders)
        