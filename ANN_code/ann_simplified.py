# Additions / suggestions 

# Shrinks the conv filters (32→16) and dense layers (128→32)
# Boosts L2 weight decay to 0.005 and dropout to 0.2
# Filter counts: 32 → 16 (vs. your original 320→160→48)
# Residual + SE: each block learns both local features and channel‑wise attention while retaining a skip connection

import tensorflow as tf
from keras import layers, Model
from keras.regularizers import l2
from keras.api.losses import Huber

def se_block(input_tensor, reduction=16):
    """Squeeze-and‑Excite block."""
    channels = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling1D()(input_tensor)
    se = layers.Dense(channels // reduction,
                      activation='relu',
                      kernel_regularizer=l2(0.005))(se)
    se = layers.Dense(channels,
                      activation='sigmoid',
                      kernel_regularizer=l2(0.005))(se)
    se = layers.Reshape((1, channels))(se)
    return layers.Multiply()([input_tensor, se])

def residual_se_block(x, filters, kernel_size, pool_size):
    """Conv → BN → ReLU → SE → pool with a residual skip."""
    shortcut = layers.Conv1D(filters, 1, padding='same',
                             kernel_regularizer=l2(0.005))(x)
    shortcut = layers.BatchNormalization()(shortcut)
    
    out = layers.Conv1D(filters, kernel_size, padding='same',
                        kernel_regularizer=l2(0.005))(x)
    out = layers.BatchNormalization()(out)
    out = layers.Activation('relu')(out)
    
    out = se_block(out)
    out = layers.MaxPooling1D(pool_size)(out)
    
    # If spatial dims differ by pooling, downsample shortcut
    if pool_size > 1:
        shortcut = layers.MaxPooling1D(pool_size)(shortcut)
    
    return layers.Add()([shortcut, out])

def def_model_se(shape1):
    inp = layers.Input(shape=(shape1, 1), name='input')
    
    # Two residual+SE blocks
    x = residual_se_block(inp, filters=32, kernel_size=5, pool_size=2)
    x = residual_se_block(x,   filters=16, kernel_size=3, pool_size=2)
    
    x = layers.Flatten()(x)
    
    # Smaller dense stack with stronger regularization
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=l2(0.005))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(32, activation='relu',
                     kernel_regularizer=l2(0.005))(x)
    x = layers.Dropout(0.2)(x)
    
    out = layers.Dense(6, activation='linear', name='layer_out')(x)
    
    model = Model(inp, out, name="simplified_res_se_cnn")
    model.compile(
        optimizer='adam',
        loss=Huber(delta=1),
        metrics=['mae']
    )
    return model

# Example usage:
# model = def_model_se(shape1)
# model.summary()


