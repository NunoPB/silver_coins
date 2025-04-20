import pandas as pd
import os
from os.path import join, splitext
import dill as pickle

'''
Reading and normalizing data. Necessary data is:
    - Two CSVs with information on real and synthetic data, 'x_in_data.csv' and 'x_out_data.csv', respectively.
      -Each CSV will include the names of the data files, the target values for each element, as well as the 'use' of each file. 
      -In the case of the synthetic files, all will be labelled as 'synth' in the csv. 
      -For the real data, each file will have one of three 'use' labels in the CSV: 'unused', 'train' and 'error'. 
      -The 'error' files are not to be used.
      -The 'unused' files will be used as the test set, and the 'train' files will be joined with the 'synth' files for training.
    - Two directories, one containing the synthetic files, and the other containing the real files, 
      information about which are found in their respective CSVs.

Normalization used is min-max normalization (feature scaling).

Output:
    -'minmax.pkl': Tuple containing two lists, mins and maxs. Each list contains six values, 
      which are either the minimum or maximum of the target values of each element, used to normalize the target values. 
      These are needed to 'unnormalize' the output data of the model.
    -'input_norm.pkl': Tuple containing two elements, the minimum and maximum of all channels of all input data, used to normalize it. 
      Here we already normalize the test set, but for unseen data these values should be used to normalize the new input.
    -'data.pkl': Tuple containing the test and training sets, which are ready to be used for training and testing.
'''

elements = ['Cu', 'Ag', 'Au', 'Hg', 'Pb', 'Bi']

path_data  = join('silver', 'i_silver', 'i_in_data.csv')
path_x_data = join('silver', 'i_silver', 'i_data')
path_synth = join('silver', 'i_silver', 'i_out_data.csv')
path_synth_data = join('silver', 'i_silver', 'i_synth')

def read_meta(path_data: str, path_synth: str) -> pd.DataFrame:
    """Read meta data of the data files.

    Combines real and synthetic target values into a single DataFrame for normalization.

    Args:
        path_data (str): path to the csv for the real data.
        path_synth (str): path to the csv for the synthetic data.

    Raises:
        ValueError: If the columns read are out of range of the provided CSVs.

    Returns:
        pd.DataFrame: DataFrame containing target values for normalization.
    """
    try:
        # list of column indexes to read, first two are the use and name of files, last 6 the elements.
        cols = [0, 1, 14, 16, 17, 18, 19, 20]
        d_data = pd.read_csv(path_data, sep=';', header=None, usecols=cols)
        d_synth = pd.read_csv(path_synth, sep=';', header = None, usecols=cols)
        d_total = pd.concat([d_data, d_synth], axis=0)
        d_total.columns = ['use', 'file'] + elements
        d_total.file = d_total.file.apply(lambda x: x.split(".")[0])
        d_total.file = d_total.file.apply(lambda x: x.split("\\")[-1])
    except ValueError as e:
        print(e)

    return d_total


def norm_meta(d_total: pd.DataFrame) -> pd.DataFrame:
    """Normalize meta data using min-max normalization (feature scaling). 

    Args:
        d_total (pd.DataFrame): DataFrame containing target values to normalize.

    Raises:
        AssertionError: If the minimum and maximum of any given element are equal, to prevent 0 division.

    Returns:
        pd.DataFrame: Normalized target values.
    """
    # Data of 'error' files is already removed to ensure it does not influence normalization.
    d_train = d_total[~d_total['use'].isin(['unused'])]
    d_test = d_total[d_total['use'].isin(['unused'])]

    mins = []
    maxs = []
    for col in elements:
        min = d_train[col].min()
        max = d_train[col].max()
        assert min != max, f'{col} 0 division'
        mins.append(min)
        maxs.append(max)
        d_train[col] = (d_train[col] - min) / (max - min)

    d_total = pd.concat([d_test, d_train], axis=0)
    with open('data\\minmax.pkl', 'wb') as file:
        pickle.dump((mins, maxs), file)

    return d_total


def read_data(path: str, error_index: list = None, channels: int = 512) -> pd.DataFrame:
    """Read data files.

    'error' files are filtered out.

    Args:
        path (str): path to the directory containing data files.
        error_index (list, optional): List of indexes of error files to remove. Defaults to None.
        channels (int, optional): Number of expected channels. Defaults to 512.

    Raises:
        AssertionError: If incorrect file extension or incorrect number of channels.

    Returns:
        pd.DataFrame: DataFrame containing sample data.
    """
    files = []
    for _, _, filenames in os.walk(path):
        for file in filenames:
            _, ext = splitext(file)
            assert ext =='.dat', f'{file} does not have .dat extention'
            files.append(file)
    
    if error_index:
        files = [file for i, file in enumerate(files) if i not in error_index]

    data = {}

    for f in files:
        file = pd.read_csv(join(path, f),header=None).iloc[:,-1]
        assert file.shape[-1] == channels, f'{f} has {file.shape} channels'
        data[f] = file.apply(lambda x: x.split()[-1]).reset_index().iloc[:,-1].astype('float')

    keys = list(data.keys())
    data = pd.DataFrame.from_dict(data).T.astype('float')
    data.index=[x.split(".")[0] for x in keys]
    data = data.sort_index(ascending=True)
    print('shape of data is ', data.shape)

    return data


def norm_data(data_test: pd.DataFrame, data_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normalize sample data using min-max normalization (feature scaling).

    Test data is normalized using training set normalization constants.

    Args:
        data_test (pd.DataFrame): Test data
        data_train (pd.DataFrame): Train data

    Raises:
        AssertionError: If minimum is equal to maximum across all training data, to prevent 0 division.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing normalized test and train sets.
    """
    # Both data_test and data_train contain their corresponding meta data, which occupy the last 8 columns.
    input_min = data_train.iloc[:, :-8].min().min()
    input_max = data_train.iloc[:, :-8].max().max()
    assert input_max != input_min, '0 division'

    data_train.iloc[:, :-8] = (data_train.iloc[:, :-8] - input_min) / (input_max - input_min)
    data_test.iloc[:, :-8] = (data_test.iloc[:, :-8] - input_min) / (input_max - input_min)

    with open('data\\input_norm.pkl', 'wb') as file:
        pickle.dump((input_min, input_max), file)

    return (data_test, data_train)


d_total = read_meta(path_data, path_synth)
# filter out data on 'error' files if any, while getting their indexes to filter out the corresponding files
mask = d_total['use'].isin(['error'])
if mask.any():
    error_index = list(d_total[mask].index)
    d_total = d_total[~mask]
else:
    error_index = None
d_total = norm_meta(d_total)

synth_data = read_data(path_synth_data)
# using error_index to filter out 'error' files
d_data = read_data(path_x_data, error_index=error_index)

d_data.columns = synth_data.columns
data = pd.concat([synth_data, d_data], axis=0)  
data['temp'] = data.index
data = data.drop_duplicates(['temp'])
data = data.drop(columns=['temp'])

data = pd.merge(data, d_total, left_index=True, right_on='file')

data_test = data[data['use'].isin(['unused'])]
data_train = data[~data['use'].isin(['unused'])]

data_test, data_train = norm_data(data_test, data_train)

with open('data\\data.pkl', 'wb') as file:
    pickle.dump((data_test, data_train), file)
