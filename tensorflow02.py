import tensorflow as tf

matrix1 = tf.constant([[2,3]])
matrix2 = tf.constant([[4],[2]])

product = tf.matmul(matrix1,matrix2)

#method1 

sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

print('====================')

with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
