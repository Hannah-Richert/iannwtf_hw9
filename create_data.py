from re import split
import tensorflow as tf
import numpy as np
#import argparse
import os
import urllib



def preprocess(ds):
    """
    Preparing our data for our model.
      Args:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: the dataset we want to preprocess

      Returns:
        - ds <tensorflow.python.data.ops.dataset_ops.PrefetchDataset>: preprocessed dataset
    """
    # casting and reshaping
    ds = ds.map(lambda image: (tf.cast(image, tf.float32)))
    ds = ds.map(lambda image: (tf.reshape(image,[28,28,1])))
    # perfornm -1, 1 min max normalization
    ds = ds.map(lambda image: ((image/128)-1))

    # shuffle, batch, prefetch our dataset
    ds = ds.shuffle(2500)
    ds = ds.batch(64,drop_remainder=True)
    ds = ds.prefetch(20)
    return ds

def load_data():

    categories = [line.rstrip(b'\n') for line in urllib.request.urlopen('https://raw.githubusercontent.com/googlecreativelab/quickdraw-dataset/master/categories.txt')]
    category = 'candle'

    # Creates a folder to download the original drawings into.
    if not os.path.isdir('npy_files'):
        os.mkdir('npy_files')

    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
    urllib.request.urlretrieve(url, f'npy_files/{category}.npy')

    images = np.load(f'npy_files/{category}.npy')
    print(f'{len(images)} images to train on')


    train_imgs = images[:30000]
    valid_imgs = images[30000:40000]
    test_imgs = images[40000:50000]

    train_ds = tf.data.Dataset.from_tensor_slices(train_imgs)
    valid_ds = tf.data.Dataset.from_tensor_slices(valid_imgs)
    test_ds = tf.data.Dataset.from_tensor_slices(test_imgs)
    # performing preprocessing steps
    train_ds = preprocess(train_ds)
    valid_ds = preprocess(valid_ds)
    test_ds = preprocess(test_ds)

    return train_ds,valid_ds,test_ds
