
import tensorflow as tf
import numpy as np
import keras

cifar10 = tf.keras.datasets.cifar10.load_data()

LAYER1 = 3072
LAYER2 = 1024
LAYER3 = 256
LAYER4 = 10

EPOCHS = 100
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 100

##############################################

W1 = tf.Variable(tf.random_uniform(shape=[LAYER1 + 1, LAYER2]) * (2 * 0.12) - 0.12)
W2 = tf.Variable(tf.random_uniform(shape=[LAYER2 + 1, LAYER3]) * (2 * 0.12) - 0.12)
W3 = tf.Variable(tf.random_uniform(shape=[LAYER3 + 1, LAYER4]) * (2 * 0.12) - 0.12)

##############################################

X = tf.placeholder(tf.float32, [None, LAYER1])
A1 = tf.concat([X, tf.ones([tf.shape(X)[0], 1])], axis=1)

Y2 = tf.matmul(A1, W1)
A2 = tf.concat([tf.sigmoid(Y2), tf.ones([tf.shape(X)[0], 1])], axis=1)

Y3 = tf.matmul(A2, W2)
A3 = tf.concat([tf.sigmoid(Y3), tf.ones([tf.shape(X)[0], 1])], axis=1)

Y4 = tf.matmul(A3, W3)
A4 = tf.sigmoid(Y4)

##############################################

ANS = tf.placeholder(tf.float32, [None, LAYER4])
D4 = tf.subtract(A4, ANS)
D3 = tf.multiply(tf.matmul(D4, tf.transpose(W3)), tf.multiply(A3, tf.subtract(1.0, A3)))
D2 = tf.multiply(tf.matmul(D3[:, :-1], tf.transpose(W2)), tf.multiply(A2, tf.subtract(1.0, A2)))

G3 = tf.matmul(tf.transpose(A3), D4)
G2 = tf.matmul(tf.transpose(A2), D3[:, :-1])
G1 = tf.matmul(tf.transpose(A1), D2[:, :-1])

W3 = W3.assign(tf.subtract(W3, tf.scalar_mul(1e-4, G3)))
W2 = W2.assign(tf.subtract(W2, tf.scalar_mul(1e-4, G2)))
W1 = W1.assign(tf.subtract(W1, tf.scalar_mul(1e-4, G1)))

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
    sess.run([W1, W2, W3], feed_dict={X: x_train[start:end], ANS: y_train[start:end]})
    print(ii)

correct_prediction = tf.equal(tf.argmax(A4,1), tf.argmax(ANS,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: x_test, ANS: y_test}))












