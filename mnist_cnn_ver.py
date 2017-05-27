import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

image_size = 28
batch_size = 100
num_labels = 10

learning_rate = 0.1


data_input = tf.placeholder(tf.float32, [None, image_size**2])
data_label = tf.placeholder(tf.float32, [None, num_labels])

keep_prob = tf.placeholder('float32')
#sess = tf.Session(config=tf.ConfigProto(device_count = {'CPU': 0}))  
sess = tf.Session()

def weight_variable(shape, name_idx):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name='weight' + str(name_idx))

def bias_variable(shape, name_idx):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name='bias' + str(name_idx))

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")
    
def mnist_conv(x, num_classes, keep_prob):

    filter1 = weight_variable([5, 5, 1, 32], 1)
    bias1 = bias_variable([32], 1)
    
    x_ = tf.reshape(x, [-1, 28, 28, 1])
    
    relu1 = tf.nn.relu(conv2d(x_, filter1) + bias1)
    pool1 = max_pool_2x2(relu1)
    
    filter2 = weight_variable([5, 5, 32, 64], 2)
    bias2 = bias_variable([64], 2)

    relu2 = tf.nn.relu(conv2d(pool1, filter2) + bias2)
    pool2 = max_pool_2x2(relu2)

    fc3 = weight_variable([7 * 7 * 64, 1024], 3)
    bias3 = bias_variable([1024], 3)

    flat_pool2 = tf.reshape(pool2, [-1, 7 * 7 * 64])
    relu3 = tf.nn.relu(tf.matmul(flat_pool2, fc3) + bias3)
    drop = tf.nn.dropout(relu3, keep_prob)

    fc4 = weight_variable([1024, 10], 4)
    bias4 = bias_variable([10], 4)

    output = tf.matmul(drop, fc4) + bias4

    return tf.nn.softmax(output)

"""
weights1 = tf.Variable(tf.truncated_normal([image_size**2,1024]))
biases1 = tf.Variable(tf.zeros([1024]))


weights2 = tf.Variable(tf.truncated_normal([1024, num_labels]))
biases2 = tf.Variable(tf.zeros([num_labels]))
hidden1 = tf.matmul(data_input, weights1) + biases1
logits = tf.matmul(hidden1, weights2) + biases2

train_pred = tf.nn.softmax(logits)
"""
logits = mnist_conv(data_input, 10, keep_prob)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=data_label))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(data_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for step in range(10000):
  
    batch = mnist.train.next_batch(batch_size)

    _, train_acc = sess.run([optimizer, accuracy], feed_dict={data_input: batch[0], data_label: batch[1], keep_prob:0.5})
    
    
    if step == 0 or (step+1) % 100 == 0:
      test_acc = sess.run(accuracy, feed_dict={data_input: mnist.test.images, data_label: mnist.test.labels, keep_prob:1.0})
      print("Iter: %4d, test_accuracy: %0.4f" %(step+1, test_acc))


    
