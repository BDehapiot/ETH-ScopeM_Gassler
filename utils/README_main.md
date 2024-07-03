## Usage

### `preextract.py`
Read data from `remote_path`, format and save to `data_path`

- Parameters
```bash
- exclude             # PKL file containing all data 
- target_pixel_size   # Tracked objects areas
```

### `main.py`
Read data from `data_path` and execute the [main procedure](#main-procedure)

- Parameters
```bash
- target              # "all"  = all dat
- threshAll           # PKL file containing all data 
- threshOut           # Tracked objects areas
- threshBod           # Tracked objects C2 intensities 
- min_size            # Tracked objects length  
- min_roundness       # Tracked objects roundness
```

## Main procedure

## Outputs

### Images
```bash
- C1_proj.tif     # Channel 1 (spores) std-projection
- C2_proj.tif     # Channel 2 (bacteria) sum-projection
- display.tif     # Tracked objects display
- labels.tif      # Tracked objects labels
- composite.tif   # C1_proj + C2_proj + display
```

### Data
```bash
- data.pkl        # PKL file containing all data 
- area.csv        # Tracked objects areas
- intensity.csv   # Tracked objects C2 intensities 
- length.csv      # Tracked objects length  
- roundness.csv   # Tracked objects roundness
- plot.jpg        # All data plot
```