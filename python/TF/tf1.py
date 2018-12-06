import tensorflow as tf

x = tf.Variable(initial_value=3.0)
y = tf.cos(x) #gradient=-sin(x)

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(y)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)
    writer.close()