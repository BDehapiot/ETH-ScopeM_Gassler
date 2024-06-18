# ETH-ScopeM_Gassler
Analysis of fungi spore germination

## Comments

### To be fixed

Done (but need to be carefully checked)
- Channel #2 should be save in 16bits (0 - 4096)
- Channel #2 needs background subtraction

Not done
- Data from `20240613_tg_24TG10_Medium` cannot be preextracted

--

- Bleaching for channel #2
    - pdR05H_20240403_tg_003 
- Unstable field of view
    - expl : 
- Bubbles & weird issues
    - pdR10H_20240403_tg_001
    - pdR10H_20240403_tg_003
- predBod issues
    - pdR10H_20240403_tg_002
    - pdR20H_20240403_tg_003

## To do list
- Segmentation
    - Measure object elongation (done)
    - Remove early touching objects (to do)
    - Remove border touching objects (done)

- Results
    - Output CSV (done)
    - Object ID outputs & plots (done)
    - Save plots as JPG (done)

- Post-analysis
    - Manually reject objects from analysis (to do)

