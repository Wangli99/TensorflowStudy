import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def addLayer(inputs,inputSize,outputSize,activationFunction=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([inputSize,outputSize]),name='W')
        Bias = tf.Variable(tf.zeros([1,outputSize]) + 0.1,name='B')
        WxPlusB = tf.add(tf.matmul(inputs,Weights),Bias,name='WxPlusB')
        if activationFunction == None:
            outputs = WxPlusB
        else:
            outputs = activationFunction(WxPlusB)
        return outputs
#Create my data
xData = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.01,xData.shape)
yData = np.square(xData) - 0.5 + noise

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(xData,yData)
plt.ion()

plt.show()
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='xInput')
    ys = tf.placeholder(tf.float32,[None,1],name='yInput')


l1 = addLayer(xs,1,10,tf.nn.relu)
prediction = addLayer(l1,10,1)
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys-prediction),1))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    for step in range(1000):
        sess.run(train_step,feed_dict={xs:xData,ys:yData})
        if step%50 == 0:
            try:
                ax.lines.remove(line[0])
            except Exception:
                pass
            print(step,sess.run(loss,feed_dict={xs:xData,ys:yData}))
            line = ax.plot(xData,sess.run(prediction,feed_dict={xs:xData}),'r',lw = 5)
            plt.pause(0.1)
        
input()