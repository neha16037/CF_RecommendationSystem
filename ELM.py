from sklearn import datasets
import numpy as np
import tensorflow as tf

iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target


print "shape of X: ", np.shape(X)
print "shape of Y: ", np.shape(y)

mean = np.mean(X)
std = np.std(X)

print "mean is: ", mean
print "std is: ", std

# R = std * np.random.randn(150, 8) + mean
R = std * np.random.randn(2, 20) + mean
print R

H = tf.nn.sigmoid(tf.matmul(X, R))
print "shape of H:", np.shape(H)

Y = np.reshape(y, (150, 1))
print "shape of Y: ", np.shape(Y)

W2 = np.linalg.lstsq(H, y)[0]
print W2