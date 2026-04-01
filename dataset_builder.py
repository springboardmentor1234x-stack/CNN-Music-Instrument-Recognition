# -------------------------------------------------------
# DATASET BUILDER
# -------------------------------------------------------

import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


CLASS_NAMES = ['cel', 'cla', 'flu', 'gac', 'gel', 'org', 'pia', 'sax', 'tru', 'vio', 'voi']


# -------------------------------------------------------
# AUGMENTATION — improved for instrument classification
# -------------------------------------------------------
def augment_spectrogram(spectrogram, label):

    # 1. Gaussian noise
    noise       = tf.random.normal(shape=tf.shape(spectrogram), mean=0.0, stddev=0.02)
    spectrogram = spectrogram + noise

    # 2. Time masking — mask up to 20% of time frames
    t       = tf.shape(spectrogram)[0]
    t_mask  = tf.random.uniform((), 0, t // 5, dtype=tf.int32)
    t_start = tf.random.uniform((), 0, t - t_mask, dtype=tf.int32)
    t_vec   = tf.cast((tf.range(t) < t_start) | (tf.range(t) >= t_start + t_mask), tf.float32)
    spectrogram = spectrogram * tf.reshape(t_vec, [-1, 1, 1])

    # 3. Frequency masking — mask up to 20% of freq bins
    f       = tf.shape(spectrogram)[1]
    f_mask  = tf.random.uniform((), 0, f // 5, dtype=tf.int32)
    f_start = tf.random.uniform((), 0, f - f_mask, dtype=tf.int32)
    f_vec   = tf.cast((tf.range(f) < f_start) | (tf.range(f) >= f_start + f_mask), tf.float32)
    spectrogram = spectrogram * tf.reshape(f_vec, [1, -1, 1])

    # 4. Random brightness (volume shift)
    spectrogram = spectrogram * tf.random.uniform((), 0.75, 1.25)

    # 5. Random time shift — roll spectrogram left or right
    shift       = tf.random.uniform((), -10, 10, dtype=tf.int32)
    spectrogram = tf.roll(spectrogram, shift, axis=0)

    # 6. Random channel dropout — zero out one delta channel occasionally
    drop_channel = tf.random.uniform((), 0, 3, dtype=tf.int32)
    mask = tf.tensor_scatter_nd_update(
        tf.ones([3], dtype=tf.float32),
        [[drop_channel]],
        [tf.cast(tf.random.uniform(()) > 0.15, tf.float32)]
    )
    spectrogram = spectrogram * tf.reshape(mask, [1, 1, 3])

    return spectrogram, label


# -------------------------------------------------------
# NORMALISATION (zero mean, unit variance per sample)
# -------------------------------------------------------
def normalise(spectrogram, label):
    mean, variance = tf.nn.moments(spectrogram, axes=[0, 1, 2])
    spectrogram    = (spectrogram - mean) / (tf.sqrt(variance) + 1e-7)
    return spectrogram, label


# -------------------------------------------------------
# MIXUP AUGMENTATION
# -------------------------------------------------------
def apply_mixup(train_ds, alpha=0.2, shuffle_buffer=500):
    ds1 = train_ds.repeat()
    ds2 = train_ds.repeat().shuffle(shuffle_buffer)

    def mixup_fn(batch1, batch2):
        x1, y1 = batch1
        x2, y2 = batch2
        lam