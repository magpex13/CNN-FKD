import tensorflow as tf
import numpy as np
import csv
import gc
import matplotlib.pyplot as plt
from pandas.io.parsers import read_csv

sample = 7049

def load():
    with open('training.csv') as file:
        csvRead = csv.reader(file)
        next(csvRead)
        dataset = np.array([next(csvRead) for _ in range(sample)])
        #dataset = np.array(list(csv.reader(file)))
        #dataset = dict(csv.DictReader(file))

    images = [np.reshape(np.fromstring(item, sep = ' '),[96,96]) for item in dataset[:, 30]]
    keypoints = dataset[:,:-1].astype(np.float32)
    # image = plt.imshow(images[1])
    # image = plt.imshow(np.reshape(np.fromstring(images[0], sep = ' '),[96,96]))
    # image = plt.imshow(np.reshape(np.fromstring(dataset[1][30], sep = ' '),[96,96]))
    # plt.show(image)
    images = np.array([ i / 255.0 for i in images]).astype(np.float32)
    keypoints = np.array([ i / 93.8988 for i in keypoints]).astype(np.float32) # 93.8988 es el valor más grande de todos los keypoints
    return images,keypoints

def loadPantas():
    dataset = read_csv('training.csv')
    dataset = dataset.dropna()
    #images = [np.fromstring(item, sep = ' ') for item in dataset['Image']]
    images = np.vstack(dataset['Image'].apply(lambda img : np.fromstring(img, sep = ' ')).values) / 255.0
    images = images.astype(np.float32)
    keypoints = dataset[dataset.columns[:-1]].apply(lambda keys : keys / 93.8988).values.astype(np.float32) # 93.8988 es el valor más grande de todos los keypoints
    return images,keypoints

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def deepnn(x):
    x_image = tf.reshape(x,[-1,96,96,1])
    wconv1 = weight_variable([3,3,1,32])
    bconv1 = bias_variable([32])
    hconv1 = tf.nn.relu(conv2d(x_image,wconv1) + bconv1)
    pconv1 = max_pool_2x2(hconv1)

    wconv2 = weight_variable([3,3,32,64])
    bconv2 = bias_variable([64])
    hconv2 = tf.nn.relu(conv2d(pconv1,wconv2) + bconv2)
    pconv2 = max_pool_2x2(hconv2)

    wconv3 = weight_variable([3,3,64,128])
    bconv3 = bias_variable([128])
    hconv3 = tf.nn.relu(conv2d(pconv2,wconv3) + bconv3)
    pconv3 = max_pool_2x2(hconv3)

    pconv3_new = tf.reshape(pconv3,[-1,12*12*128])
    wfc1 = weight_variable([12*12*128,500])
    bfc1 = bias_variable([500])
    hfc1 = tf.nn.relu(tf.matmul(pconv3_new,wfc1) + bfc1)

    wfc2 = weight_variable([500,30])
    bfc2 = bias_variable([30])
    hfc2 = tf.nn.softmax(tf.matmul(hfc1,wfc2) + bfc2)

    return hfc2

images,keypoints = loadPantas()
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

y_ = deepnn(x)
#cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_))
loss = tf.reduce_mean(tf.pow(y - y_,2))
train_step = tf.train.RMSPropOptimizer(0.0001,decay = 1e-8).minimize(loss)
# correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch = 8
epochs = 15
epochGroup = int(sample / batch)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for ep in range(epochs):
        print("Epoch N° : %d" % (ep + 1))
        for i in range(epochGroup):
            idx = int( i % (epochGroup))
            train_step.run(feed_dict={x: images[idx * batch: (idx + 1) * batch], y: keypoints[idx * batch: (idx + 1) * batch]})
            if i % int(epochGroup / 10) == 0:
                train_error = loss.eval(feed_dict={
                    x: images[idx * batch: (idx + 1) * batch], y: keypoints[idx * batch: (idx + 1) * batch]})
                print('     step %d, training error %f' % (i, train_error))

    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #     x: mnist.test.images, y: mnist.test.labels}))

gc.collect()