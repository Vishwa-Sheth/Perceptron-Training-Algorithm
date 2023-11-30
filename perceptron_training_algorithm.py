import numpy as np
import matplotlib.pyplot as plt
import copy

# Setting up the random seed as constant to get the same results everytime, so that we can do the analysis
np.random.seed(1)
def neuralNetwork(w, S):

  # Getting the output for S using weights and step activation function
  Y = np.heaviside(np.dot(S,w),1)
  S1 = []
  S0 = []

  # Diving S into S1 and S0 as per the output obtained
  for i in range(0,Y.size):
    if Y[i] == 0:
      S0.append(S[i])
    else:
      S1.append(S[i])

  S1 = np.array(S1)
  S0 = np.array(S0)

  return S1, S0, Y

# Implementation of perceptron training algorithm for all etas
def perceptron_training_algorithm(eta, S, Y, w_prime,w_optimal,S1,S0):
  epoch_array = []
  miss_classification_array = []
  w_array = []
  for i in range(0,len(eta)):

    # Plotting the optimal boundary and S0,S1 points
    plt.figure(figsize=(10,6))
    plt.scatter(S1[:,1], S1[:,2], color='blue', label = "Samples with output 1 (S1)", marker='s')
    plt.scatter(S0[:,1], S0[:,2], color='red', label = "Samples with output 0 (S0)", marker='.')
    x = [-1, 1]
    y = [-(w_optimal[0] - w_optimal[1])/w_optimal[2], -(w_optimal[0] + w_optimal[1])/w_optimal[2]]
    plt.plot(x, y, color = 'blue', label = 'Optimal boundary : %.3f + %.3f*X1 + %.3f*X2 = 0' %(w_optimal[0],w_optimal[1],w_optimal[2]))

    epoch  = 0
    w = copy.copy(w_prime)
    miss_classification = []

    #Predicting the Y by w' weights
    Y_pred = np.heaviside(np.dot(S,w),1)
    miss = np.sum(Y != Y_pred)
    miss_classification.append(miss)

    # Untill Y != Y_pred
    while(Y != Y_pred).any():
      epoch = epoch + 1

      # For every x1,x2 if the y_pred != y then updating the weights
      for j in range(S.shape[0]):
        y = np.heaviside(w[0]*S[j][0] + w[1]*S[j][1] + w[2]*S[j][2], 1)
        if(Y[j] != y ):
          w = w + (Y[j] - y)*eta[i]*S[j]
      # Predicting the Y after updating the weights
      Y_pred = np.heaviside(np.dot(S,w),1)

      # Counting the number of misclassifications obtained by using the updated weights
      miss = np.sum(Y != Y_pred)
      miss_classification.append(miss)

    # Storing the values of epoch, miss_classification and w for each eta
    epoch_array.append(epoch)
    miss_classification_array.append(miss_classification)
    w_array.append(w)

    # Plotting the boundary line created by predicted weights
    x = [-1, 1]
    y = [-(w[0] - w[1])/w[2], -(w[0] + w[1])/w[2]]
    print("Predicted weights are ", w)

    plt.plot(x, y, label = 'Boundary with eta = %.1f : %.3f + %.3f*X1 + %.3f*X2 = 0' %(eta[i],w[0],w[1],w[2]))
    plt.title("Sample points, optimal boundary line and boundary lines with eta = %0.1f" %(eta[i]))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc='upper right')
    plt.show()
  return epoch_array, miss_classification_array, w_array

# The first method called with given samples array and weights w
def main_method(samples_array, w):
  for s in range(len(samples_array)):

    # Initialising Sample set S
    samples = samples_array[s]
    print("Samples taken : %0.0f" %samples)

    S = np.ones([samples, 3], dtype=float)
    S[:,1] = np.random.uniform(-1,1,samples)
    S[:,2] = np.random.uniform(-1,1,samples)

    # Diving S into S0 and S1 according to the output
    S1, S0, Y = neuralNetwork(w, S)


    # Plotting all S1 and S0 points
    plt.figure(figsize=(10,6))
    plt.scatter(S1[:,1], S1[:,2], color='blue', label = "Samples with output 1 (S1)", marker='s')
    plt.scatter(S0[:,1], S0[:,2], color='red', label = "Samples with output 0 (S0)", marker='.')

    # Plotting the optimal boundary line w0 + w1x1 + w2x2 = 0
    x = [-1, 1]
    y = [-(w[0] - w[1])/w[2], -(w[0] + w[1])/w[2]]
    plt.plot(x, y, color = 'blue', label = 'Optimal boundary : %.3f + %.3f*X1 + %.3f*X2 = 0' %(w[0],w[1],w[2]))
    plt.title("Sample points and optimal boundary for %0.0f Samples" %samples)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend(loc='upper right')
    plt.show()


    # Initializing w' for all etas
    w_prime = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)])
    print("W_Prime : ", w_prime)
    eta = [1,10,0.1]

    # Calling the perceptron training algorithm and getting epoch_array, miss_classification_array and w_array
    epoch_array, miss_classification_array, w_array = perceptron_training_algorithm(eta, S, Y, w_prime, w,S1,S0)

    plt.figure(figsize=(10,6))

    # Plotting Epoch number vs The number of misclassifications for all etas
    for k in range(len(eta)):
      print("No of epoch of eta = %0.1f is %0.0f" %(eta[k],epoch_array[k]))
      print("Miss_classifications  = ",miss_classification_array[k])
      epoch = np.array(list(range(0,epoch_array[k] + 1)))
      plt.plot(epoch, miss_classification_array[k], label = 'Eta = %.1f' %eta[k])
      plt.title("Epoch number vs The number of misclassifications for %0.0f Samples on Eta = %.1f" %(samples,eta[k]))
      plt.xlabel("Epoch number")
      plt.ylabel("The number of misclassifications")
      plt.legend(loc='upper right')
      plt.show()

# initializing samples array and weights
samples_array = [100,1000]
w = np.ones([3], dtype=float)
w[0] = np.random.uniform(-0.25,0.25)
w[1] = np.random.uniform(-1,1)
w[2] = np.random.uniform(-1,1)
print("Optimal weights : ",w)

main_method(samples_array, w)