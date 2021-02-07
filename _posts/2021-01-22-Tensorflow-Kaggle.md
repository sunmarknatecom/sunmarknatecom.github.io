---
layout: post
title: tensorflow and kaggle
categories: [tensorflow, kaggle]
tags: [cassava]
published: true	
---	
from https://www.kaggle.com/jessemostipak/getting-started-tpus-cassava-leaf-disease

```bash
 >>> import math, re, os
 >>> import tensorflow as tf
 >>> import numpy as np
 >>> import pandas as pd
 >>> import matplotlib.pyplot as plt
 >>> from kaggle_datasets import KaggleDatasets
 >>> from tensorflow import keras
 >>> from functools import partial
 >>> from sklearn.model_selection import train_test_split
 >>>
 >>> print("Tensorflow version " + tf.__version__)
 Tensorflow version 2.2.0
```

```bash
 >>> try:
 ...     tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
 ...     print('Device:', tpu.master())
 ...     tf.config.experimental_connect_to_cluster(tpu)
 ...     tf.tpu.experimental.initialize_tpu_system(tpu)
 ...     strategy = tf.distribute.experimental.TPUStrategy(tpu)
 >>> except:
 ...     strategy = tf.distribute.get_strategy()
 >>> print('Number of replicas:', strategy.num_replicas_in_sync)
 Device: grpc://10.0.0.2:8470
 Number of replicas: 8
```

```bash
 >>> AUTOTUNE = tf.data.experimental.AUTOTUNE
 >>> GCS_PATH = KaggleDatasets().get_gcs_path()
 >>> BATCH_SIZE = 16 * strategy.num_replicas_in_sync
 >>> IMAGE_SIZE = [512, 512]
 >>> CLASSES = ['0', '1', '2', '3', '4']
 >>> EPOCHS = 25
```

```bash
 >>> def decode_image(image):
 ...     image = tf.image.decode_jpeg(image, channels=3)
 ...     image = tf.cast(image, tf.float32) / 255.0
 ...     image = tf.reshape(image, [*IMAGE_SIZE, 3])
 ...     return image
```

```bash
 >>> def read_tfrecord(example, labeled):
 ...    tfrecord_format = {
 ...         "image": tf.io.FixedLenFeature([], tf.string),
 ...         "target": tf.io.FixedLenFeature([], tf.int64)
 ...     } if labeled else {
 ...         "image": tf.io.FixedLenFeature([], tf.string),
 ...         "image_name": tf.io.FixedLenFeature([], tf.string)
 ...     }
 ...     example = tf.io.parse_single_example(example, tfrecord_format)
 ...     image = decode_image(example['image'])
 ...     if labeled:
 ...         label = tf.cast(example['target'], tf.int32)
 ...         return image, label
 ...     idnum = example['image_name']
 ...     return image, idnum
```

```bash
 >>> def load_dataset(filenames, labeled=True, ordered=False):
 ...     ignore_order = tf.data.Options()
 ...     if not ordered:
 ...         ignore_order.experimental_deterministic = False # disable order, increase speed
 ...     dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files
 ...     dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
 ...     dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
 ...     return dataset
```

```bash
 >>> TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
 ...    tf.io.gfile.glob(GCS_PATH + '/train_tfrecords/ld_train*.tfrec'),
 ...    test_size=0.35, random_state=5
 ... )
 >>> 
 >>> TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/test_tfrecords/ld_test*.tfrec')
```

```bash
 >>> def data_augment(image, label):
 ...    # Thanks to the dataset.prefetch(AUTO) statement in the following function this happens essentially for free on TPU. 
 ...    # Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
 ...    image = tf.image.random_flip_left_right(image)
 ...    return image, label
```
```bash
 >>> def get_training_dataset():
 ...     dataset = load_dataset(TRAINING_FILENAMES, labeled=True)  
 ...     dataset = dataset.map(data_augment, num_parallel_calls=AUTOTUNE)  
 ...     dataset = dataset.repeat()
 ...     dataset = dataset.shuffle(2048)
 ...     dataset = dataset.batch(BATCH_SIZE)
 ...     dataset = dataset.prefetch(AUTOTUNE)
 ...     return dataset
```
```bash

```
