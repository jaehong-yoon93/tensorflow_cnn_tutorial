import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

image_size = 28
batch_size = 100
num_labels = 10

learning_rate = 0.1


data_input = tf.placeholder(tf.float32, [None, image_size**2])
data_label = tf.placeholder(tf.float32, [None, num_labels])

#sess = tf.Session(config=tf.ConfigProto(device_count = {'CPU': 0}))  
sess = tf.Session()


weights1 = tf.Variable(tf.truncated_normal([image_size**2,1024]))
biases1 = tf.Variable(tf.zeros([1024]))


weights2 = tf.Variable(tf.truncated_normal([1024, num_labels]))
biases2 = tf.Variable(tf.zeros([num_labels]))

sess.run(tf.global_variables_initializer())  

hidden1 = tf.matmul(data_input, weights1) + biases1
logits = tf.matmul(hidden1, weights2) + biases2
train_pred = tf.nn.softmax(logits)


loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=data_label))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

  

correct_prediction = tf.equal(tf.argmax(train_pred, 1), tf.argmax(data_label, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#  sess.run(tf.global_variables_initializer())

for step in range(10000):
  
    batch = mnist.train.next_batch(batch_size)

    _, train_acc = sess.run([optimizer, accuracy], feed_dict={data_input: batch[0], data_label: batch[1]})
    
    
    if step == 0 or (step+1) % 100 == 0:
      test_acc = sess.run(accuracy, feed_dict={data_input: mnist.test.images, data_label: mnist.test.labels})
      print("Iter: %4d, test_accuracy: %0.4f" %(step+1, test_acc))


    
