import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def addLayer(inputs, inputSize, outputSize, activationFunction=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([inputSize, outputSize]), name='Weights')
        Bias = tf.Variable(tf.zeros([1, outputSize])+0.1, name='Bias')
        WxPlusB = tf.add(tf.matmul(inputs, Weights), Bias, name='WxPlusB')
        if activationFunction == None:
            return WxPlusB
        else:
            return activationFunction(WxPlusB)

#用与计算每一步模型评价的准确性
def compute_accuracy(v_xs, v_xy):
    global xs, ys
    global prediction
    
    correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_xy})
    return result

#得到数据
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None,784])
ys = tf.placeholder(tf.float32, [None,10])


#创建神经网络
prediction = addLayer(xs, 784, 10, tf.nn.softmax)

#caculate loss
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), 1))
tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

#初始化参数
init = tf.initialize_all_variables()

#开始训练
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/', sess.graph)
    merged = tf.summary.merge_all()
    os.system("cls")
    print('=========================================================')
    for step in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
        if step % 50 == 0:
            final_accuracy = compute_accuracy(mnist.test.images, mnist.test.labels)
            #tf.summary.scalar('accuracy', final_accuracy)
            test = sess.run(merged, feed_dict={xs:batch_xs, ys:batch_ys})
            writer.add_summary(test, step)
            print(final_accuracy)
    
    
#test usage
