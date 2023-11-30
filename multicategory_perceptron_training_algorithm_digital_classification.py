import numpy as np
import matplotlib.pyplot as plt
import copy

# %matplotlib inline

np.random.seed(1)

import os
import gzip

DATA_URL = 'http://yann.lecun.com/exdb/mnist/'

def load_data():
    x_tr = load_images('train-images-idx3-ubyte.gz')
    y_tr = load_labels('train-labels-idx1-ubyte.gz')
    x_te = load_images('t10k-images-idx3-ubyte.gz')
    y_te = load_labels('t10k-labels-idx1-ubyte.gz')

    return x_tr, y_tr, x_te, y_te

def load_images(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 28 * 28) / np.float32(256)

def load_labels(filename):
    maybe_download(filename)
    with gzip.open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

# Download the file, unless it's already here.
def maybe_download(filename):
    if not os.path.exists(filename):
        from urllib.request import urlretrieve
        print("Downloading %s" % filename)
        urlretrieve(DATA_URL + filename, filename)

Xtrain, ytrain, Xtest, ytest = load_data()


def perceptron_training_algo(eta,threshold,n,epoch,w,Xtrain,ytrain,limit):
  errors = []
  while(limit >= len(errors)):
    error = 0
    desired_output = np.zeros([ ytrain.size, 10])
    for i in range(n):
      v = np.dot(w,Xtrain[i])
      if(ytrain[i] != np.argmax(v)):
        error += 1
      desired_output[i][ytrain[i]] = 1

    errors.append(error)
    epoch += 1
    for i in range(n):
      w =  w + np.dot((eta*(desired_output[i] - np.heaviside(np.dot(w,Xtrain[i]),1)).reshape(10,1)), Xtrain[i].reshape(1,784))
    if((error/n) <= threshold):
      break
  return errors,w

def applying_perceptron_training_algo(w, Xtest, ytest):
  error = 0
  for i in range(ytest.size):
    v = np.dot(w,Xtest[i])
    if(ytest[i] != np.argmax(v)):
        error += 1
  return error


eta = 1
threshold = 0

# For n = 50, eta = 1 and threshold = 0
n = 50
epoch = 0
w_prev = np.array(np.random.uniform(-1,1,size = [10,784]))
w = copy.copy(w_prev)
errors_train,w_new = perceptron_training_algo(eta,threshold,n,epoch,w,Xtrain,ytrain,n)

epoch_number = np.arange(0, len(errors_train), 1, dtype=int)
plt.plot(epoch_number, errors_train, label = 'n=50')
plt.title("Epoch number vs The number of misclassifications for n = 50, eta = 1 and threshold = 0")
plt.xlabel("Epoch number")
plt.ylabel("The number of misclassifications")
plt.legend(loc='upper right')
plt.show()

errors_test = applying_perceptron_training_algo(w_new, Xtest, ytest)
print("The percentage of misclassified test samples for n = 50, eta = 1 and threshold = 0 is ",(errors_test/ytest.size)*100,"%")


# For n = 1000, eta = 1 and threshold = 0
n = 1000
epoch = 0
w = copy.copy(w_prev)
errors_train, w_new = perceptron_training_algo(eta,threshold,n,epoch,w,Xtrain,ytrain,n)

epoch_number = np.arange(0, len(errors_train), 1, dtype=int)
plt.plot(epoch_number, errors_train, label = 'n=1000')
plt.title("Epoch number vs The number of misclassifications for n = 1000, eta = 1 and threshold = 0")
plt.xlabel("Epoch number")
plt.ylabel("The number of misclassifications")
plt.legend(loc='upper right')
plt.show()

errors_test = applying_perceptron_training_algo(w_new, Xtest, ytest)
print("The percentage of misclassified test samples for n = 1000, eta = 1 and threshold = 0 is ",(errors_test/ytest.size)*100,"%")


# For n = 60000, eta = 1 and threshold = 0
n = 60000
epoch = 0
w = copy.copy(w_prev)

errors_train,w_new = perceptron_training_algo(eta,threshold,n,epoch,w,Xtrain,ytrain,50)
epoch_number = np.arange(0, len(errors_train), 1, dtype=int)
plt.plot(epoch_number, errors_train, label = 'n=60000')
plt.title("Epoch number vs The number of misclassifications for n = 60000, eta = 1 and threshold = 0 by limiting the epochs to 50")
plt.xlabel("Epoch number")
plt.ylabel("The number of misclassifications")
plt.legend(loc='upper right')
plt.show()

errors_test = applying_perceptron_training_algo(w_new, Xtest, ytest)
print("The percentage of misclassified test samples for n = 60000, eta = 1 and threshold = 0 by limiting the epochs to 50 is ",(errors_test/ytest.size)*100,"%")

# For n = 60000, eta = 1 and threshold = 0.14
for i in range(3):
  n = 60000
  epoch = 0
  threshold = 0.14
  w = np.array(np.random.uniform(-1,1, size = [10,784]))

  errors_train, w_new = perceptron_training_algo(eta,threshold,n,epoch,w,Xtrain,ytrain,n)
  # print(errors_train)
  epoch_number = np.arange(0, len(errors_train), 1, dtype=int)
  plt.plot(epoch_number, errors_train, label = 'n=60000')
  plt.title("Epoch number vs The number of misclassifications for n = 60000, eta = 1 and threshold = 0.14")
  plt.xlabel("Epoch number")
  plt.ylabel("The number of misclassifications")
  plt.legend(loc='upper right')
  plt.show()

  errors_test = applying_perceptron_training_algo(w_new, Xtest, ytest)
  print("The percentage of misclassified test samples for n = 60000, eta = 1 and threshold = 0.14 is ",(errors_test/ytest.size)*100,"%")