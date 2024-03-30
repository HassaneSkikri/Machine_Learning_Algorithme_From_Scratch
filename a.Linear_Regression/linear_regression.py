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
    


# Testing the algorithm
    
if __name__ == '__main__':

    # importing the liberaries
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import numpy as np
    import matplotlib.pyplot as plt

    # load the dataset
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    # split the dataset into train and test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    # get some information about the dataset
    def information(X,y):
        print(X.shape)
        print(y.shape)

    information(X,y)

    # mean squared error
    def MSE(y_true,y_pred):
        return np.mean((y_true-y_pred)**2)
    
    # create an instance of LinearRegression
    lr = LinearRegression(learning_rate = 0.01)

    # fit the model
    lr.fit(X_train, y_train)

    # make predictions
    y_pred = lr.predict(X_test)

    # claculate the mean squared erro 
    mse_value  = MSE(y_test, y_pred)

    # print it 
    print(mse_value)

    # plot the results
    y_pred_line = lr.predict(X)
    cmap = plt.get_cmap("viridis")
    fig = plt.figure(figsize=(8, 6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10, label="Training data")
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10, label="testing data")
    plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("linear regression algorithm" ,c="b",size=18)
    plt.legend()
    plt.show()
