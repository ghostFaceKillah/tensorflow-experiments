import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def fun(x, a=10.0, b=5.0, c=0.001):
    return a * x**2 + b * x + c

x_data = 2 * np.random.rand(200).astype("float32") - 1
y_data = fun(x_data)


# Try to find values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in xrange(201):
    sess.run(train)
    if step % 20 == 0:
        print step, sess.run(W), sess.run(b)

W_opt = sess.run(W)
b_opt = sess.run(b)

x_test = 2 * np.random.rand(200).astype("float32") - 1

y_hat = W_opt * x_test + b_opt
y_test = fun(x_test)

plt.plot(x_test, y_hat, '.', label="real function")
plt.plot(x_test, y_test, '.', label="learned function")
plt.legend()
plt.show()
