#%% Imports -------------------------------------------------------------------

import nd2

#%% Functions -----------------------------------------------------------------

def open_data(path, metadata=True):

    # Read nd2 file
    with nd2.ND2File(path) as ndfile:
        stack = ndfile.asarray()
        # nC, nZ, nY, nX = stack.shape
        # vY, vX, vZ = ndfile.voxel_size()
    
    # if metadata:
        
    #     metadata = {
    #         "nZ" : nZ, "nY" : nY, "nX" : nX, 
    #         "vZ" : vZ, "vY" : vY, "vX" : vX,
    #         }
    
        # return stack, metadata
    
    return stack    