import numpy as np

class Perceptron:

    # N: number of columns in input feature vector
    # alpha: learning rate
   def  __init__(self, N, alpha=0.01):
       # initialize the weight matrix and store the learning rate
       self.W = np.random.randn(N + 1) / np.sqrt(N)
       self.alpha = alpha

   def step(self, x):
       return 1 if(x > 0) else 0

   # X: feature matrix for data points
   # y: labels
   def fit(self, X, y, epochs=10):
       # add one to data points to make way for biases in weights matrix
       ones = np.ones((X.shape[0]))
       print(ones)
       X = np.c_[X, ones]
       print(X)
       print(self.W)
       for epoch in np.arange(0, epochs):
           for (x, target) in zip(X, y):
               print(x)
               # we multiply signal strength and weight
               # for every connection from input node to perceptron. This is a generic approach and we can use it
               # in every NN architecture
               weightedInput = np.dot(x, self.W)
               print(weightedInput)
               p = self.step(weightedInput)
               if p != target:
                   error = p - target
                   self.W += -self.alpha * error * x

   def predict(self, X, addBias=True):
       X = np.atleast_2d(X)
       if addBias:
           X = np.c_[X, np.ones((X.shape[0]))]

       return self.step(np.dot(X, self.W))









