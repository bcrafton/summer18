import tensorflow as tf
import numpy as np
import keras

###############################################

learning_rate = 1e-1

batch_size = 64

num_examples = 50000
num_test = 10000
num_epochs = 50

cifar10 = tf.keras.datasets.cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(num_examples, 32, 32, 3)
x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(num_test, 32, 32, 3)
x_test = x_test / 255.
y_test = keras.utils.to_categorical(y_test, 10)

###############################################

drop_out_flag = tf.placeholder(tf.bool)

x_train -= np.mean(x_train)
x_test -= np.mean(x_test)

x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])

y = tf.placeholder(tf.float32, [None, 10])

global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')

#x_input = tf.layers.dropout(x,rate=0.1,training = drop_out_flag) 

conv1 = tf.layers.conv2d(inputs = x, filters = 96 ,kernel_size = [5,5], padding = "same", activation = tf.nn.relu, name = 'conv1')

conv1_pool = tf.layers.max_pooling2d(inputs = conv1, pool_size= [3,3], strides = 2, padding = 'same')

conv1_input = tf.layers.dropout(conv1_pool, rate=0.25,training = drop_out_flag)
#conv layer 2
conv2 = tf.layers.conv2d(inputs = conv1_input, filters = 128, kernel_size = [5,5], padding = "same", activation = tf.nn.relu, name = 'conv2')

conv2_pool = tf.layers.max_pooling2d(inputs = conv2, pool_size = [3,3], strides = 2 , padding = 'same')

conv2_input = tf.layers.dropout(conv2_pool, rate=0.25,training = drop_out_flag)

#conv layer 3
conv3 = tf.layers.conv2d(inputs = conv2_input, filters = 256, kernel_size = [5,5], padding = "same", activation = tf.nn.relu, name = 'conv3')

conv3_pool = tf.layers.max_pooling2d(inputs = conv3, pool_size = [3,3], strides = 2 , padding = 'same')

conv3_input = tf.layers.dropout(conv3_pool,rate =0.5,training = drop_out_flag)
 
flat = tf.contrib.layers.flatten(conv3_input)

fc1 = tf.layers.dense(flat,2048,activation = tf.nn.relu)

#fc1_input = tf.layers.dropout(fc1, rate=0.5,training = drop_out_flag)


fc2 = tf.layers.dense(fc1,2048,activation = tf.nn.relu)

#fc2_input = tf.layers.dropout(fc2,rate=0.5,training = drop_out_flag)

softmax = tf.layers.dense(fc2, units=10, activation = tf.nn.softmax)

###############################################

y_pred_cls = tf.argmax(softmax,axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=softmax, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=0.9,
                                   beta2=0.999,
epsilon=1).minimize(loss, global_step=global_step) #var_list = 


#optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = 0.99).minimize#(loss, global_step = global_step)

correct_prediction = tf.equal(y_pred_cls, tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

###############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cum_acc = 0
count = 1

for e in range(num_epochs):
    for s in range(int(num_examples/batch_size)):
        batch_xs = x_train[s*batch_size:(s+1)*batch_size]
        batch_ys = y_train[s*batch_size:(s+1)*batch_size]
        
        _, _, _,batch_acc = sess.run([global_step, optimizer, loss, accuracy], feed_dict ={x:batch_xs, y: batch_ys,drop_out_flag: True})

        #cum_acc += batch_acc
        if s % 10 == 0:
            #roll_acc = cum_acc/count
            print(batch_acc, 'for epoch', e)
            
        
        count += 1

count = 1
total_acc = 0
for s in range(int(num_test/batch_size)):
    batch_xt = x_test[s*batch_size:(s+1)*batch_size]
    batch_yt = y_test[s*batch_size:(s+1)*batch_size]

    acc = sess.run(accuracy, feed_dict={x: batch_xt, y: batch_yt, drop_out_flag: False})
    total_acc += acc
    count += 1

true_acc = total_acc/count
print('*************************************')

print('Test Accuracy is:', true_acc)
