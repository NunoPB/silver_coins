import numpy as np
import pandas as pd
from keras import layers, Model
from keras.api.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.api.regularizers import l2
from keras.api.losses import Huber
import dill as pickle
import matplotlib.pyplot as plt

'''
1D CNN model:
    - 3 1D convolutional layer each with batchnormalization and maxpooling.
    - 2 dense layers with l2 regularization, each with a dropout layer.
    - ReLu activation function.
    - Final output layer with linear activation.

    - Adam optimizer.
    - Huber loss function (mix of mse and mae)

The training set is split into a training set (80%) and a validation set (20%).

When training the model, there are three callbacks:
    -checkpoint: saves best model.
    -earlystop: stops the model if no improvment.
    -reduce_lr: reduces learning rate if no improvment.

Output:
    -'best_model_x.keras': trained model.
    -'Loss_plot.png': picture of train and validation loss.
    -'history.xlsx': history of model training across epochs.
'''


elements = ['Cu', 'Ag', 'Au', 'Hg', 'Pb', 'Bi']

with open('data\\data.pkl', 'rb') as file:
    data_test, data_train = pickle.load(file)

# 8 columns of meta data
shape1 = data_train.shape[-1] - 8

def def_model():
    xinput = layers.Input(shape=(shape1, 1), name='input')
    
    conv1 = layers.Conv1D(320, 5, activation='relu')(xinput)
    bn1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling1D(pool_size=2)(bn1)
    
    conv2 = layers.Conv1D(160, 5, activation='relu')(pool1)
    bn2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling1D(pool_size=2)(bn2)

    conv3 = layers.Conv1D(48, 3, activation='relu')(pool2)
    bn3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling1D(pool_size=2)(bn3)
    
    x = layers.Flatten()(pool3)

    x = layers.Dense(320, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = layers.Dropout(0.1)(x)

    x = layers.Dense(24, activation='relu', kernel_regularizer=l2(0.002))(x)
    x = layers.Dropout(0.1)(x)

    y = layers.Dense(6, activation='linear', name='layer_out')(x)

    model = Model(xinput, y, name="tuned_model_cnn")
    model.compile(
        optimizer='adam',
        loss=Huber(delta=1), 
        metrics=['mae']
    )
    
    return model


model = def_model()
model.summary()

checkpoint_path = 'data\\best_model_i.keras'
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    mode='min',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    save_freq='epoch'
)
earlystop = EarlyStopping(
    monitor='val_loss', 
    mode='min', 
    patience=10,
    verbose=1) 
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    mode='min',
    factor=0.2,         
    patience=3,         
    min_lr=1e-6,      
    verbose=1         
)

history = model.fit(
    np.array(data_train.iloc[:,:shape1]).reshape(data_train.shape[0],shape1,1),
    data_train[elements],
    validation_split=.2,
    epochs=20,
    verbose=1,
    callbacks=[checkpoint, 
               earlystop, 
               reduce_lr
    ]
) 

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('data\\Loss_plot.png')
plt.show()

history_df = pd.DataFrame(history.history)
history_df.insert(0, 'Epoch', range(1, len(history_df) + 1))
history_df.to_excel('data\\history.xlsx', index=False)
