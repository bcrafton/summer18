
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
import os
import math
import numpy
import numpy as np
import time
from PIL import Image

batch_size = 128
num_classes = 1000
epochs = 50
data_augmentation = False

label_counter = 0

training_images = []
training_labels = []

def parse_function(filename, label):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [256, 256])
    return image, label

def train_preprocess(image, label):
    # image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

print ("building dataset")

for subdir, dirs, files in os.walk('/home/bcrafton3/ILSVRC2012/train/'):
    for folder in dirs:
        for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
            for file in folder_files:
                training_images.append(os.path.join(folder_subdir, file))
                # training_labels.append( keras.utils.to_categorical(label_counter, num_classes) )
                
                # label = np.zeros(num_classes)
                # label[label_counter] = 1
                # training_labels.append(label)

                training_labels.append(label_counter)

        label_counter = label_counter + 1
        print (str(label_counter) + "/" + str(num_classes))

filename = tf.placeholder(tf.string, shape=[None])
label_num = tf.placeholder(tf.int64, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices((filename, label_num))
dataset = dataset.shuffle(len(training_images))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
dataset = dataset.prefetch(8)

print("Data is ready...")

iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()

features = tf.reshape(features, (-1, 256, 256, 3))
labels = tf.one_hot(labels, depth=num_classes)

# features = tf.Print(features, [tf.shape(features)], message="")
# labels = tf.Print(labels, [tf.shape(labels)], message="")

#x = tf.placeholder(tf.float32, [None, 256, 256, 3])
#y = tf.placeholder(tf.float32, [None, num_classes])

conv1 = tf.layers.conv2d(inputs=features, filters=16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv1_pool = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2, padding='same')

conv2 = tf.layers.conv2d(inputs=conv1_pool, filters=16, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv2_pool = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2, padding='same')

conv3 = tf.layers.conv2d(inputs=conv2_pool, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv3_pool = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2,2], strides=2, padding='same')

conv4 = tf.layers.conv2d(inputs=conv3_pool, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv4_pool = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2,2], strides=2, padding='same')

conv5 = tf.layers.conv2d(inputs=conv4_pool, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv5_pool = tf.layers.max_pooling2d(inputs=conv5, pool_size=[2,2], strides=2, padding='same')

flat = tf.contrib.layers.flatten(conv5_pool)
#flat = tf.Print(flat, [tf.shape(flat)], message="")

#spec = tf.layers.InputSpec(dtype=tf.float32, shape=(128, 2048), ndim=2, max_ndim=2, min_ndim=2)
fc1 = tf.layers.dense(inputs=flat, units=2048, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1, units=2048, activation=tf.nn.relu)
fc3 = tf.layers.dense(inputs=fc2, units=num_classes)

predict = tf.argmax(fc3, axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=labels))
correct = tf.equal(predict, tf.argmax(labels, 1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1).minimize(loss)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

sess.run(iterator.initializer, feed_dict={filename: training_images, label_num: training_labels})

for i in range(0, epochs):
    for j in range(0, len(training_images), batch_size):
        print (j)
        if ((j % 1024) == 0):
            corr = sess.run([total_correct])
            print (corr)
            # sess.run([optimizer])
        else: 
            sess.run([optimizer])
    
    print('epoch {}/{}'.format(i, epochs))

