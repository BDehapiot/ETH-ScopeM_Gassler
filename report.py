#%% Imports -------------------------------------------------------------------

import napari
from skimage import io
from pathlib import Path
from functions import get_all, get_outlines

#%% Inputs --------------------------------------------------------------------

# Paths
report_path = Path(Path.cwd(), "report")
target = "pdINDw_20240131_tg_002"
   
#%%

mask_all = io.imread(report_path / (target + "_C1_mask-all.tif"))
mask_bodies = io.imread(report_path / (target + "_C1_mask-bodies.tif"))
mask_all_edm = get_all(mask_all)
mask_all_outlines = get_outlines(mask_all)

io.imsave(
    report_path / (target + "_C1_mask-bodies_edm.tif"),
    mask_all_edm.astype("float32"), check_contrast=True,
    )
io.imsave(
    report_path / (target + "_C1_mask-bodies_outlines.tif"),
    mask_all_outlines.astype("float32"), check_contrast=True,
    )


