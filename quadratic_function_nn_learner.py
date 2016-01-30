import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def fun(x, a=10.0, b=5.0, c=0.001):
    return a * x**2 + b * x + c

def get_batch_of_data():
    x_data = 4 * np.random.rand(500, 1).astype("float32") - 2
    y_data = fun(x_data)
    return x_data, y_data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

x = tf.placeholder("float", shape=[None, 1])
y_ = tf.placeholder("float", shape=[None, 1])

S = 100
W_in = weight_variable([1, S])
b_in = weight_variable([S])

mid = tf.nn.relu(tf.matmul(x, W_in) + b_in)

W_out = weight_variable([S, 1])
b_out = weight_variable([1])

y = tf.matmul(mid, W_out) + b_out

loss = tf.reduce_mean(tf.square(y - y_))
train = tf.train.GradientDescentOptimizer(0.005).minimize(loss)
"""
It's really interesting to see how important the learning rate is.
For x = 0.005 it is optimal
for x = 0.05  the gradient is way too much and we jump around solutions
for x = 0.0001 it takes ages to converge
for x = 0.5 there is some kind of exploding gradient error
"""


sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

for step in xrange(10000):
    x_data, y_data = get_batch_of_data()
    train.run(feed_dict={x:x_data, y_: y_data})
    if step % 500 == 0:
        print step
        x_test, y_test = get_batch_of_data()
        y_hat = y.eval(feed_dict={x: x_test})

        plt.plot(x_test, y_hat, '.', label="real function")
        plt.plot(x_test, y_test, '.', label="learned function")
        plt.legend()
        plt.title("after {} iteration".format(step))
        plt.savefig("img/func_{}.png".format(step))
        plt.close()
