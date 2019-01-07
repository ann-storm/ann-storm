from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip

import numpy
import scipy.io as sio

from six.moves import xrange  # pylint: disable=redefined-builtin

import base
from tensorflow.python.framework import dtypes


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images_from_mat(file_name, var_name='var'):
    """Extract the images from .mat format."""
    print('Extracting', file_name)
    mat_contents = sio.loadmat(file_name)
    data = mat_contents[var_name]
    return data


def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels_from_mat(file_name, var_name='var'):
    print('Extracting', file_name)
    mat_contents = sio.loadmat(file_name)
    labels = mat_contents[var_name]
    return labels


class DataSet(object):
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float64,
                 reshape=True):
        """Construct a DataSet.
        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.
        """
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float64):
            raise TypeError('Invalid image dtype %r, expected uint8 or float64' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float64:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float64)
                images = numpy.multiply(images, 1.0 / 1.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 169
            if self.one_hot:
                fake_label = [1] + [0] * 1
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)
            ]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float64,
                   reshape=False,
                   validation_size=1000):
    # type: (object, object, object, object, object, object) -> object
    if fake_data:
        def fake():
            return DataSet([], [], fake_data=True, one_hot=one_hot, dtype=dtype)

        train = fake()
        validation = fake()
        test = fake()
        return base.Datasets(train=train, validation=validation, test=test)

    DATA_DIR = '../training_data/storm_color/'

    TRAIN_IMAGES = DATA_DIR + 'train_data_storm_twocolor.mat'
    TRAIN_LABELS = DATA_DIR + 'train_label_storm_twocolor.mat'

    train_images = extract_images_from_mat(TRAIN_IMAGES)
    train_labels = extract_labels_from_mat(TRAIN_LABELS)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'
                .format(len(train_images), validation_size))

    validation_images = train_images[20000:]
    validation_labels = train_labels[20000:]
    train_images = train_images[:20000] 
    train_labels = train_labels[:20000]

    train = DataSet(train_images, train_labels, dtype=dtype, reshape=reshape)
    validation = DataSet(validation_images,
                         validation_labels,
                         dtype=dtype,
                         reshape=reshape)
    test = DataSet(validation_images, validation_labels, dtype=dtype, reshape=reshape)

    return base.Datasets(train=train, validation=validation, test=test)


