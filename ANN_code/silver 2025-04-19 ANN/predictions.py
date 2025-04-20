import pandas as pd
import numpy as np
from keras.api.models import load_model
from keras import Model
import dill as pickle

'''
Testing model on test set. Model is also tested on training set, but this is for validation only.

Output:
    -'test_x.xlsx': predictions on test set. 
    -'train_x.xlsx': predicitons on train set.
'''

elements = ['Cu', 'Ag', 'Au', 'Hg', 'Pb', 'Bi']

with open('data\\data.pkl', 'rb') as file:
    data_test, data_train = pickle.load(file)

with open('data\\minmax.pkl', 'rb') as file:
    mins, maxs = pickle.load(file)

# 8 columns of meta data
shape1 = data_test.shape[-1] - 8

def keras_model(model_keras: Model) -> tuple[np.ndarray, np.ndarray]:
    """Do predictions on test and training sets.

    Use saved normalization constants to unnormalize model outputs.

    Args:
        model_keras (Model): Trained model.

    Returns:
        tuple[np.ndarray, np.ndarray]: Predictions on test and train sets.
    """
    model = load_model(model_keras)
    predictions1 = model.predict(np.array(data_test.iloc[:,:shape1]).reshape(data_test.shape[0],shape1,1))
    predictions2 = model.predict(np.array(data_train.iloc[:,:shape1]).reshape(data_train.shape[0],shape1,1))
    for i in range(len(elements)):
        predictions1[:, i] = predictions1[:, i] * (maxs[i] - mins[i]) + mins[i]
        predictions2[:, i] = predictions2[:, i] * (maxs[i] - mins[i]) + mins[i]

    print(f'test predictions shape: {predictions1.shape}')
    print(f'train predictions shape: {predictions2.shape}')

    return (predictions1, predictions2)


def save_preds(preds: np.ndarray, data: pd.DataFrame, data_type: str, data_version: str) -> None:
    """Generates excel file containing the final 'unnormalized' output.
    Also contains original target values for easy comparison.

    Args:
        preds (np.ndarray): model output.
        data (pd.DataFrame): data containing target values.
        data_type (str): If train data we need to unnormalize the target values aswell.
        data_version (str): version of data.
    """
    df = pd.DataFrame(preds, columns=elements)

    file_name = data['file'].reset_index(drop=True)
    use = data['use'].values
    df.insert(0, 'use', use)
    df.insert(1, 'file', file_name)

    empty_column = pd.Series([''] * len(df), name='')

    target_data = data[elements].reset_index(drop=True)
    target_data.columns = elements
    if data_type == 'train':
        for i in range(len(elements)):
            target_data.iloc[:, i] = target_data.iloc[:, i] * (maxs[i] - mins[i]) + mins[i]
    
    df = pd.concat([df, empty_column, target_data], axis=1)
    df.to_excel(f'data\\{data_type}_{data_version}.xlsx', index=False)


predictions1, predictions2 = keras_model('data\\best_model_i.keras')
save_preds(predictions1, data_test, 'test', 'i')
save_preds(predictions2, data_train, 'train', 'i')
