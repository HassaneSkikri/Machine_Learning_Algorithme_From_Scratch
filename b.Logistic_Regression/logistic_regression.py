import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        n_samples,n_features = X.shape
        self.weights = np.zeros(n_features) # initialize the weights with array of zeros
        self.bias = 0 # initialize the bias with 0

        # gradient descent

        for _ in range(self.n_iters):
            # first claculate the linear model
            linear_model = np.dot(X,self.weights) + self.bias

            # y_prediction
            y_prediction = self._segmoid(linear_model)

            # calculate the derivatives
            dw = (1/n_samples) * np.dot(X.T,(y_prediction-y))
            db = (1/n_samples) * np.sum(y_prediction-y)
            # update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    def predict(self,X):
        linear_model = np.dot(X,self.weights) + self.bias
        y_prediction = self._segmoid(linear_model )
        return [1 if i >0.5 else 0 for i in y_prediction]
    def _segmoid(self,x):
        return 1/(1+np.exp(-x))


if __name__ == "__main__":
    #import liberaries
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import numpy as np

    bc = datasets.load_breast_cancer()
    X,y = bc.data,bc.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    def accuracy(y_true,y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    regressor = LogisticRegression(learning_rate=0.0001,n_iters=1000)
    regressor.fit(X_train,y_train)
    predictions = regressor.predict(X_test)

    print("Logistic Regression classification accuracy : ",accuracy(y_test,predictions))