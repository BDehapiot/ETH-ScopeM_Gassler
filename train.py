#%% Imports -------------------------------------------------------------------

import napari
import random
import numpy as np
from skimage import io
from pathlib import Path
import albumentations as A
import matplotlib.pyplot as plt
import segmentation_models as sm
from joblib import Parallel, delayed 

# TensorFlow
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Functions
from functions import get_skel, get_patches

#%% Inputs --------------------------------------------------------------------

# Path
train_path = Path(Path.cwd(), 'data', 'train')

# Patches
downscale_factor = 4
size = 512 // downscale_factor
overlap = size // 8

# Data augmentation
random.seed(42)
iterations = 1000

# Train model
validation_split = 0.2
n_epochs = 100
batch_size = 16

#%% Pre-processing ------------------------------------------------------------

img_patches, msk_patches, skl_patches = [], [], []
for path in train_path.iterdir():
    if 'mask' in path.name:
        
        # Open data
        msk = io.imread(path)
        img = io.imread(str(path).replace('_mask', ''))
        
        # Get skeleton
        skl = get_skel(msk).astype("float32")
        msk = (msk > 0).astype("float32")
                
        # Extract patches
        img_patches.append(get_patches(img, size, overlap))
        msk_patches.append(get_patches(msk, size, overlap))
        skl_patches.append(get_patches(skl, size, overlap))

img_patches = np.stack([patch for patches in img_patches for patch in patches])
msk_patches = np.stack([patch for patches in msk_patches for patch in patches])
skl_patches = np.stack([patch for patches in skl_patches for patch in patches])

# Display 
viewer = napari.Viewer()
viewer.add_image(img_patches)
viewer.add_image(msk_patches) 
viewer.add_image(skl_patches) 
            
#%% Augmentation --------------------------------------------------------------

# augment = True if iterations > 0 else False

# if augment:
    
#     # Define augmentation operations
#     operations = A.Compose([
#         A.VerticalFlip(p=0.5),              
#         A.RandomRotate90(p=0.5),
#         A.HorizontalFlip(p=0.5),
#         A.Transpose(p=0.5),
#         A.GridDistortion(p=0.5),
#         ], additional_targets={'mask2': 'mask'}
#         )

#     # Augment data
#     def augment_data(images, masks1, masks2, operations):      
#         idx = random.randint(0, len(images) - 1)
#         outputs = operations(
#             image=images[idx,...], 
#             mask=masks1[idx,...], 
#             mask2=masks2[idx,...]
#             )
#         return outputs['image'], outputs['mask'], outputs['mask2']
    
#     # Execute parallel augmentation for each iteration
#     outputs = Parallel(n_jobs=-1)(
#         delayed(augment_data)(img_patches, msk_patches, skl_patches, operations)
#         for i in range(iterations)
#         )
#     img_patches = np.stack([data[0] for data in outputs])
#     msk_patches = np.stack([data[1] for data in outputs])
#     skl_patches = np.stack([data[2] for data in outputs]) 
    
#     # Display 
#     viewer = napari.Viewer()
#     viewer.add_image(img_patches)
#     viewer.add_image(msk_patches) 
#     viewer.add_image(skl_patches)

#%% Model training ------------------------------------------------------------

# Define & compile model
model = sm.Unet(
    'resnet34', 
    input_shape=(None, None, 1), 
    classes=1, 
    activation='sigmoid', 
    encoder_weights=None,
    )
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', 
    metrics=['mse']
    )

# Checkpoint & callbacks
model_checkpoint_callback = ModelCheckpoint(
    filepath="model_weights.h5",
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True
    )
callbacks = [
    EarlyStopping(patience=20, monitor='val_loss'),
    model_checkpoint_callback
    ]

# train model
history = model.fit(
    x=img_patches, y=skl_patches,
    validation_split=validation_split,
    batch_size=batch_size,
    epochs=n_epochs,
    callbacks=callbacks,
    )

# Plot training results
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()