# Tutorial for implementation of CNN with tensorflow
import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from .ops import *
from scipy import ndimage, misc

class convNet(object):
    def __init__(self, config, sess, train_data, test_data):

        self.sess = sess

        self.train_data = train_data
        self.test_data = test_data
        self.train_range = config.train_range
        self.test_range = config.test_range

        # training detail
        self.batch_size = config.batch_size
        self.g_step = tf.Variable(0)
        self.lr = tf.train.exponential_decay(config.lr, self.g_step, len(self.train_range), 0.95)
        self.image_size = config.image_size
        self.channels = config.channels
        self.num_classes = config.num_classes
        self.max_epoch = config.max_epoch

        self.input_data = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, self.channels])
        self.input_labels = tf.placeholder(tf.float32, [None, self.num_classes])

    def build_model(self):
        """
        Write down your code here for what you want to make
        Find logits and optimize
        WANRNING: end point operation should be appended to op_list
        """
        with tf.name_scope('conv1') as scope:
            kernel = tf.Variable(tf.truncated_normal([11, 11, 3, 16], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(self.input_data, kernel, [1, 4, 4, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(bias, name=scope)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        
        with tf.name_scope('conv2') as scope:
            kernel = tf.Variable(tf.truncated_normal([5, 5, 16, 32], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(bias, name=scope)

        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
        norm2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

        with tf.name_scope('conv3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 32, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(norm2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv3 = tf.nn.relu(bias, name=scope)

        with tf.name_scope('conv4') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv4 = tf.nn.relu(bias, name=scope)

        with tf.name_scope('conv5') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 32], dtype=tf.float32, stddev=1e-1), name='weights')
            conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, shape=[32], dtype=tf.float32), trainable=True, name='biases')
            bias = tf.nn.bias_add(conv, biases)
            conv5 = tf.nn.relu(bias, name=scope)

        pool5 = tf.nn.max_pool(conv5, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool5')

        shape = pool5.get_shape().as_list()
        reshape = tf.reshape(pool5, [-1, shape[1] * shape[2] * shape[3]])

        with tf.name_scope('fc6') as scope:
            fc6_weight = tf.Variable(tf.truncated_normal([shape[1] * shape[2] * shape[3], 256], dtype=tf.float32, stddev=1e-1), name='weights')
            fc6_biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
            fc6 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, fc6_weight), fc6_biases))

        with tf.name_scope('fc7') as scope:
            fc7_weight = tf.Variable(tf.truncated_normal([256, self.num_classes], dtype=tf.float32, stddev=1e-1), name='weights')
            fc7_biases = tf.Variable(tf.constant(0.0, shape=[self.num_classes], dtype=tf.float32), trainable=True, name='biases')
            fc7 = tf.nn.bias_add(tf.matmul(fc6, fc7_weight), fc7_biases)

        logits = fc7

        op_list = []

        self.preds = tf.nn.softmax(logits) #
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.input_labels))

        regularizers = (tf.nn.l2_loss(fc6_weight) + tf.nn.l2_loss(fc6_biases) +
                  tf.nn.l2_loss(fc7_weight) + tf.nn.l2_loss(fc7_biases))
        # Add the regularization term to the loss.
        self.loss += 5e-4 * regularizers

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 40)
        optimizer = tf.train.AdamOptimizer(self.lr, name = 'adam_opt')
        optimize = optimizer.apply_gradients(zip(grads, tvars))

        op_list.append(optimize) #
        with tf.control_dependencies(op_list):
            self.op = tf.no_op(name='no_op')

    def train(self):
        total_preds = []
        total_labels = []
        total_loss = []

        for step, (data_in, label_in) in tqdm(enumerate(self.data_iteration(self.train_data, True)), desc = 'train'):
            _, _, loss, preds = self.sess.run([self.op, self.g_step, self.loss, self.preds],
                            feed_dict = {
                                    self.input_data: data_in,
                                    self.input_labels: label_in
                            })
            total_preds.append(preds)
            total_labels.append(label_in)
            total_loss.append(loss)

        total_loss = sum(total_loss)
        total_preds = np.concatenate(total_preds, axis = 0)
        total_labels = np.concatenate(total_labels, axis = 0)
        return total_loss, accuracy(total_preds, total_labels)

    def test(self):
        total_preds = []
        total_labels = []
        total_loss = []

        for step, (data_in, label_in) in tqdm(enumerate(self.data_iteration(self.test_data, False)), desc = 'test'):
            loss, preds = self.sess.run([self.loss, self.preds],
                            feed_dict = {
                                    self.input_data: data_in,
                                    self.input_labels: label_in
                            })
            total_preds.append(preds)
            total_labels.append(label_in)
            total_loss.append(loss)

        total_loss = sum(total_loss)
        total_preds = np.concatenate(total_preds, axis = 0)
        total_labels = np.concatenate(total_labels, axis = 0)
        return total_loss, accuracy(total_preds, total_labels)


    def run(self, task_name):
        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()

        for i in range(self.max_epoch):
            train_loss, train_acc = self.train()
            test_loss, test_acc = self.test()
            print("Epoch: %d, Train loss: %.3f, Train Acc: %.3f" % (i+1, train_loss, train_acc))
            print("Epoch: %d, Test loss: %.3f, Test Acc: %.3f" % (i+1, test_loss, test_acc))
            if not os.path.exists("checkpoints"):
                os.mkdir("checkpoints/")
            save_path = self.saver.save(self.sess, "checkpoints/model.ckpt")

        test_loss, test_acc = self.test()
        print("Task: %s, Test loss: %.3f, Test Acc: %.3f" % (str(task_name), test_loss, test_acc))

    def demo(self, file_path):
        tf.initialize_all_variables().run()
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, "checkpoints/model.ckpt")

        im = ndimage.imread(file_path)
        im = misc.imresize(im, (224, 224))
        from PIL import Image
        import matplotlib.pyplot as plt
        #plt.imshow(im)
        #plt.show()

        in_im = np.reshape(im, (1, 224, 224, 3))

        pred = self.sess.run([self.preds], feed_dict = { self.input_data: in_im })

        label_dict = {
            0: 'daisy',
            1: 'dandelion',
            2: 'roses',
            3: 'sunflowers',
            4: 'tulips'
        }

        print("prediction:" + label_dict[np.argmax(pred[0])])
        plt.imshow(im)
        plt.show()


    def data_iteration(self, data, is_train = True):
        """
        implement additional data preprocessing.
        this code is just before the data input

        """

        data_range = None
        if is_train:
            data_range = self.train_range
        else:
            data_range = self.test_range
        batch_len = len(data_range) // self.batch_size

        for l in xrange(batch_len):
            b_idx = data_range[self.batch_size * l:self.batch_size * (l+1)]

            batch_inputs = np.zeros((self.batch_size, self.image_size, self.image_size, self.channels), np.float32)
            batch_labels = np.zeros((self.batch_size), np.int32)

            for b in range(self.batch_size):
                image, label = np.copy(data[b_idx[b]])
                batch_inputs[b, :, :, :] = image
                batch_labels[b] = label

            batch_labels = (np.arange(self.num_classes) == batch_labels[:, None]).astype(np.float32)
            yield batch_inputs, batch_labels
