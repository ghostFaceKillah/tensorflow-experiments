import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


x_data = np.random.rand(100).astype("float32")
y_data = x_data * 0.1 + 0.3

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

x_test = np.random.rand(100).astype("float32")

y_hat = W_opt * x_test + b_opt
y_test = 0.1 * x_test + 0.3

plt.plot(x_test, y_hat, label="real function")
plt.plot(x_test, y_test, label="learned function")
plt.legend()
plt.show()


