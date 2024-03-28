import numpy as np
class LinearRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    def fit(self,X,y):
        # first init parameters
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features) # initialize the weights with array of zeros
        self.bias = 0 # initialize the bias with 0
        # then train the model
        # check first the formula of the gradient descent algorithm to understand
        # how it works
        for _ in range(self.n_iters):
            # first claculate the approximation
            # y_pred = wx + b
            y_pred = np.dot(X,self.weights) + self.bias
            # compute gradients

            # calculate the derivative 
            # 1/N * sum[i=1:n] (2*X(i)*(y_pred-y(i)))
            dw = (1/n_samples)* np.dot(X.T,(y_pred - y))
            # 1/N * sum[i=1:n] (2*(y_pred-y(i)))
            db = (1/n_samples)* np.sum((y_pred - y))


            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self, X):
        y_approximated = np.dot(X, self.weights) + self.bias
        return y_approximated
    