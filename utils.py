import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf

#------ Methods and Classes for parsing Images ----#
def load_classes(train_path):
    classes = []
    path = os.path.join(train_path, '*')
    files = glob.glob(path)
    for f in files:
        classes.append(f[len(train_path)+1:])
    return classes

def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []
    cls = []

    for fields in classes:
        index = classes.index(fields)
        print('Reading {} file (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            img_names.append(flbase)
            cls.append(fields)
    images = np.array(images)
    labels = np.array(labels)
    img_names = np.array(img_names)
    cls = np.array(cls)

    return images, labels, img_names, cls


class DataSet(object):
  def __init__(self, images, labels, img_names, cls):
    self.num_examples = images.shape[0]

    self.images = images
    self.labels = labels
    self.img_names = img_names
    self.cls = cls
    self.epochs_done = 0
    self.index_in_epoch = 0

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self.index_in_epoch
    self.index_in_epoch += batch_size

    if self.index_in_epoch > self.num_examples:
      # After each epoch we update this
      self.epochs_done += 1
      start = 0
      self.index_in_epoch = batch_size
      assert batch_size <= self.num_examples
    end = self.index_in_epoch

    return self.images[start:end], self.labels[start:end], self.img_names[start:end], self.cls[start:end]

def read_train_sets(train_path, image_size, classes, validation_size):
    class DataSets(object):
        pass
    data_sets = DataSets()

    images, labels, img_names, cls = load_train(train_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    data_sets.train_data = DataSet(train_images, train_labels, train_img_names, train_cls)
    data_sets.valid_data = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    return data_sets

def read_test_set(test_path, image_size, classes):
    images, labels, img_names, cls = load_train(test_path, image_size, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)
    data_sets = DataSet(images, labels, img_names, cls)
    return data_sets

#------- Convolutional Network Methods ------#
def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    ## define the weights
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## creating biases
    biases = create_biases(num_filters)
    ## creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME') + biases
    ## activation function
    layer = tf.nn.relu(layer)
    ## max-pooling
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
