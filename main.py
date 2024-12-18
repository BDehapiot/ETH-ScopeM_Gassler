#%% Imports -------------------------------------------------------------------

from pathlib import Path

# Functions
from functions import batch
    
#%% Inputs --------------------------------------------------------------------

# Paths
# remote_path = Path(r"\\scopem-idadata.ethz.ch\BDehapiot\remote_Gassler")
remote_path = Path("D:\local_Gassler")
data_path = Path(remote_path, "data")
# target = "all" 
target = "pdINDw_20240604_tg_012"
# target = "mmR20n_20240613_tg_008"
overwrite = True
   
# Parameters
threshAll = 0.05 #
threshOut = 0.25 #
threshBod = 0.05 #
min_size = 32 # 
min_roundness = 0.3 #

# Patches
downscale_factor = 4
size = 512 // downscale_factor
overlap = size // 2

#%% Execute -------------------------------------------------------------------

if __name__ == "__main__":
    
    if target == "all":
        paths = list(data_path.glob("*_C1.tif"))
        iPlot, iDisplay = False, False
    else:
        paths = [data_path / (target + "_C1.tif")]
        iPlot, iDisplay = True, True
        
    for i, path in enumerate(paths):
        exp_name = path.name.replace("_C1.tif", "")
        test_path = data_path / exp_name / path.name.replace("_C1.tif", "_data.pkl")
        if not test_path.is_file() or overwrite:            
            batch(
                path, 
                ["area", "length", "roundness", "intensity"],
                size, overlap,
                threshAll, threshOut, threshBod, 
                min_size, min_roundness,
                iPlot=iPlot, iDisplay=iDisplay,
                )