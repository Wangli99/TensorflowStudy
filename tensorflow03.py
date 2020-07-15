import tensorflow as tf

state = tf.Variable(0,name='counter')
one = tf.constant(1)

newValue = tf.add(state,one)
update = tf.assign(state,newValue)
second = tf.add(update,one)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        result = sess.run(second)
        print(result)