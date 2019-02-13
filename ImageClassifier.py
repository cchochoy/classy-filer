from utils import *
from utils import DataSet
import tensorflow as tf
import numpy as np

import os,glob,cv2
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class ImageClassifier(object):

    def __init__(self, train_path, classifier_name='my_classifier', image_size=64, validation_size=0.2, batch_size=100):
        self.train_path = train_path
        self.classes = load_classes(train_path)
        self.num_classes = len(self.classes)
        self.batch_size = batch_size
        self.validation_size = validation_size
        self.image_size = image_size
        self.num_channels = 3
        self.total_iterations = 0
        self.classifier_name = classifier_name
        self.data = read_train_sets(self.train_path, self.image_size, self.classes, self.validation_size)
        print("Number of files in the Training-set:\t\t{}".format(len(self.data.train_data.labels)))
        print("Number of files in the Validation-set:\t{}".format(len(self.data.valid_data.labels)))

    def train(self, num_iteration):

        #------- Variable Init ----------#
        self.total_iterations = 0
        # Placeholder
        x = tf.placeholder(tf.float32, shape=[None, self.image_size,self.image_size, self.num_channels], name='x')
        y = tf.placeholder(tf.float32, (None, self.num_classes), name="y")

        #------- Create Convolution Network -----#
        layer_conv1 = create_convolutional_layer(x, self.num_channels, 8, 32)
        layer_conv2 = create_convolutional_layer(layer_conv1, 32, 5, 64)
        layer_conv3= create_convolutional_layer(layer_conv2, 64, 5, 128)
        layer_conv4= create_convolutional_layer(layer_conv3, 128, 5, 256)
        layer_conv5= create_convolutional_layer(layer_conv4, 256, 5, 215)
        layer_flat = create_flatten_layer(layer_conv3)
        layer_fc1 = create_fc_layer(layer_flat, layer_flat.get_shape()[1:4].num_elements(), 128, True)
        logits = create_fc_layer(layer_fc1, 128, self.num_classes, False)
        print(layer_conv1, layer_conv2, layer_conv3, layer_conv4, layer_conv5, layer_flat, layer_fc1, logits)

        #----- Predictions ------#
        ## labels
        y_true = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y_true')
        y_true_class = tf.argmax(y_true, dimension=1)
        # prediction
        y_pred = tf.nn.softmax(logits,name='y_pred')
        y_pred_class = tf.argmax(y_pred, dimension=1)

        # loss
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        # accuracy
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
        correct_prediction = tf.equal(y_pred_class, y_true_class)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        #------ Training Phase -------#
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for i in range(self.total_iterations, self.total_iterations + num_iteration):
            x_batch, y_true_batch, _, class_batch = self.data.train_data.next_batch(self.batch_size)
            x_valid_batch, y_valid_batch, _, valid_class_batch = self.data.valid_data.next_batch(self.batch_size)

            feed_dict_train = {x: x_batch, y_true: y_true_batch}
            feed_dict_validate = {x: x_valid_batch, y_true: y_valid_batch}

            session.run(optimizer, feed_dict=feed_dict_train)
            if i % int(self.data.train_data.num_examples/self.batch_size) == 0:
                val_loss = session.run(cost, feed_dict=feed_dict_validate)
                epoch = int(i / int(self.data.train_data.num_examples/self.batch_size))
                acc = session.run(accuracy, feed_dict=feed_dict_train)
                val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
                msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
                print(msg.format(epoch + 1, acc, val_acc, val_loss))
                saver.save(session, './' + self.classifier_name)

        self.total_iterations += num_iteration

    def predict(self, image_path, model=None, sess=None):
        # ----- Loading Model ----- #
        # path of the image
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = dir_path +'/' + image_path
        images = []
        # loading image
        image = cv2.imread(filename)
        image = cv2.resize(image, (self.image_size, self.image_size),0,0, cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        images = np.multiply(images, 1.0/255.0)
        # reshape to feed the network
        x_batch = images.reshape(1, self.image_size, self.image_size, self.num_channels)

        # load the corresponding model
        if model is not None :
            model.restore(sess, tf.train.latest_checkpoint('./'))
        else :
            sess = tf.Session()
            saver = tf.train.import_meta_graph(self.classifier_name + '.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./'))

        # get the graph of the model
        graph = tf.get_default_graph()

        # load the corresponding tensors from the model
        y_pred = graph.get_tensor_by_name("y_pred:0")
        x= graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, self.num_classes))

        # ---- Prediction ---- #
        # perform prediction
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)

        id = np.argmax(result)
        return self.classes[id] # label with the most probability


    def test(self, test_path):
        # ------ Testing ------- #
        total_guess = 0
        correct_guess = 0

        sess = tf.Session()
        saver = tf.train.import_meta_graph(self.classifier_name + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./'))

        for c in self.classes:
            class_path = glob.glob(test_path + '/' + c + '/*')
            for img_path in class_path:
                prediction = self.predict(img_path, model = saver, sess = sess)
                if prediction == c:
                    correct_guess += 1
                total_guess +=1

        print("--- Testing Accuracy: {}%".format(correct_guess/total_guess * 100))
