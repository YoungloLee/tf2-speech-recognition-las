import os
import csv
from sklearn.model_selection import train_test_split
from text_syllable import token_index, SOS, EOS

import numpy as np
import tensorflow as tf
from infolog import log


def dataset(metadata_filename, args):
    batch_size = args.batch_size
    buffer_size = 20000
    num_mels = args.num_mels
    max_length = 500
    input_dir = os.path.join(os.path.dirname(metadata_filename), 'inputs')
    label_dir = os.path.join(os.path.dirname(metadata_filename), 'labels')

    with open(metadata_filename, encoding='utf-8') as f:
        metadata = [line.strip().split('|') for line in f]
        metadata = [x for x in metadata if int(x[-1]) <= max_length]
        timesteps = sum([int(x[4]) for x in metadata])
        sr = args.sample_rate
        hours = timesteps / sr / 3600
        log('Loaded metadata for {} examples ({:.2f} hours)'.format(len(metadata), hours))

    with open(os.path.join('datasets/filter_train_list.csv')) as train_csv, open(
            os.path.join('datasets/filter_test_list.csv')) as test_csv:
        train_list = csv.reader(train_csv)
        valid_list = csv.reader(test_csv)
        train_list = [x[0] for x in train_list][1:]
        valid_list = [x[0] for x in valid_list][1:]

    all_list = [os.path.basename(x[0]) for x in metadata]
    train_intersect = np.in1d(all_list, train_list)
    valid_intersect = np.in1d(all_list, valid_list)

    train_meta = [x for idx, x in enumerate(metadata) if train_intersect[idx]]
    valid_meta = [x for idx, x in enumerate(metadata) if valid_intersect[idx]]

    # train_meta, valid_meta = train_test_split(metadata, test_size=0.01, shuffle=True)

    train_steps = int(np.ceil(len(train_meta) / args.batch_size))
    valid_steps = int(np.ceil(len(valid_meta) / args.batch_size))

    train_input_path = [x[-4] for x in train_meta]
    train_label_path = [x[-3] for x in train_meta]

    valid_input_path = [x[-4] for x in valid_meta]
    valid_label_path = [x[-3] for x in valid_meta]

    def encode(input_path, label_path):
        # Collect signals
        input = np.load(os.path.join(input_dir, input_path.numpy().decode('utf8'))).astype('float32')
        # instance normalization
        input = (input - input.mean()) / input.std()
        with open(os.path.join(label_dir, label_path.numpy().decode('utf8')), 'r', encoding='utf-8') as f_in:
            label = f_in.readline()
        label = np.array([token_index[SOS]] + [token_index[x] for x in label] + [token_index[EOS]]).astype('int32')
        return input, label

    def tf_encode(input_path, label_path):
        result_input, result_label = tf.py_function(encode, [input_path, label_path], [tf.float32, tf.int32])
        result_input.set_shape([None, num_mels])
        result_label.set_shape([None])
        return result_input, result_label

    def filter_max_length(x, y):
        return tf.logical_and(tf.shape(x)[0] <= max_length, tf.size(y) <= max_length)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_input_path, train_label_path))
    train_dataset = train_dataset.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.filter(filter_max_length)
    # cache the dataset to memory to get a speedup while reading from it.
    # train_dataset = train_dataset.cache()
    train_dataset = train_dataset.shuffle(buffer_size).padded_batch(batch_size, padded_shapes=([None, num_mels], [None]))
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_input_path, valid_label_path))
    valid_dataset = valid_dataset.map(tf_encode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.filter(filter_max_length).padded_batch(batch_size, padded_shapes=([None, num_mels], [None]))

    return train_dataset, valid_dataset, train_steps, valid_steps
