# Baseline removal (ALS) takes out slow drift so your net doesn’t have to learn it.
# Savitzky–Golay smoothing cuts high‑frequency noise while preserving peak shapes.
# Augmentation teaches the CNN invariance to small intensity jitters, shifts in wavelength calibration, and overall scaling—multiplying your effective dataset without extra lab runs.

import numpy as np
from scipy.signal import savgol_filter
from scipy import sparse
from scipy.sparse.linalg import spsolve

# Baseline correction via Asymmetric Least Squares
def baseline_als(y, lam=1e5, p=0.01, niter=10):
    """
    y     : raw spectrum (1D array)
    lam   : smoothness weight (higher = smoother baseline)
    p     : asymmetry penalty (0 < p < 1)
    niter : # of iterations
    returns baseline-corrected spectrum
    """
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        Z = W + lam * D.dot(D.T)
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return y - z

# Denoising with Savitzky–Golay
def smooth_sg(y, window_length=11, polyorder=3):
    """
    window_length : must be odd, < len(y)
    polyorder     : <= window_length-1
    """
    return savgol_filter(y, window_length, polyorder)

# Spectral augmentations
def augment_spectrum(y,
                     noise_std=0.005,
                     shift_max=3,
                     scale_range=(0.9, 1.1)):
    """
    noise_std   : fraction of max(y) to use as Gaussian noise σ
    shift_max   : max number of bins to circularly shift the spectrum
    scale_range : (low, high) factor to multiply all intensities
    """
    y_aug = y.copy()
    # (a) additive noise
    σ = noise_std * np.max(y)
    y_aug += np.random.randn(*y.shape) * σ

    # (b) random wavelength shift
    shift = np.random.randint(-shift_max, shift_max + 1)
    y_aug = np.roll(y_aug, shift)

    # (c) random intensity scaling
    scale = np.random.uniform(*scale_range)
    y_aug *= scale

    return y_aug

# Putting it all together in a tf.data pipeline
import tensorflow as tf

def preprocess_and_augment(X, y, training=True):
    """
    X: 1D numpy spectrum
    y: label vector
    """
    # baseline and smoothing always
    X = baseline_als(X)
    X = smooth_sg(X)

    if training:
        X = augment_spectrum(X)

    # reshape for Conv1D: (length, 1)
    return X.astype(np.float32).reshape(-1, 1), y

def make_dataset(df, elements, batch_size=16, training=True):
    X = df.iloc[:, :-8].values  
    Y = df[elements].values

    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if training:
        ds = ds.shuffle(len(df), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda x, y: tf.py_function(
            func=lambda a, b: preprocess_and_augment(a, b, training),
            inp=[x, y],
            Tout=(tf.float32, tf.float32)
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# Usage:
# train_ds = make_dataset(data_train, elements, batch_size=32, training=True)
# val_ds   = make_dataset(data_train, elements, batch_size=32, training=False)
# model.fit(train_ds, validation_data=val_ds, …)
