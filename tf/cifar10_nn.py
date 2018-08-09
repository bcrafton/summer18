
import tensorflow as tf
import numpy as np
import keras
from nn_tf import nn_tf

cifar10 = tf.keras.datasets.cifar10.load_data()

LAYER1 = 3072
LAYER2 = 1024
LAYER3 = 512
LAYER4 = 256
LAYER5 = 10

EPOCHS = 1000
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 100

##############################################

W1 = tf.Variable(tf.random_uniform(shape=[LAYER1 + 1, LAYER2]) * (2 * 0.12) - 0.12)
W2 = tf.Variable(tf.random_uniform(shape=[LAYER2 + 1, LAYER3]) * (2 * 0.12) - 0.12)
W3 = tf.Variable(tf.random_uniform(shape=[LAYER3 + 1, LAYER4]) * (2 * 0.12) - 0.12)
W4 = tf.Variable(tf.random_uniform(shape=[LAYER4 + 1, LAYER5]) * (2 * 0.12) - 0.12)

X = tf.placeholder(tf.float32, [None, LAYER1])
Y = tf.placeholder(tf.float32, [None, LAYER5])

model = nn_tf(size=[LAYER1, LAYER2, LAYER3, LAYER4, LAYER5],  \
              weights=[W1, W2, W3, W4],                       \
              alpha=1e-5,                                     \
              bias=True)
# predict
predict = model.predict(X)

# train     
[W1, W2, W3, W3] = model.train(X, Y)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAIN_EXAMPLES, LAYER1)
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TEST_EXAMPLES, LAYER1)
y_test = keras.utils.to_categorical(y_test, 10)

for ii in range(0, EPOCHS * TRAIN_EXAMPLES, BATCH_SIZE):
    start = ii % TRAIN_EXAMPLES
    end = ii % TRAIN_EXAMPLES + BATCH_SIZE
    sess.run([W1, W2, W3], feed_dict={X: x_train[start:end], Y: y_train[start:end]})
    print(ii)

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))












