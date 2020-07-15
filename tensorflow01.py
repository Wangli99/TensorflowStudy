import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

xData = np.random.random(100)
yData = 0.1*xData + 0.3

Weights = tf.Variable(tf.random_uniform(shape=[1],dtype=tf.float32,minval=-1,maxval=1))
Bias = tf.Variable(tf.zeros([1]))

y = Weights*xData + Bias


loss = tf.reduce_mean(tf.square(y-yData))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(Weights),sess.run(Bias))