from sklearn.model_selection import train_test_split
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

X, y = datasets.make_regression(
    n_samples=100, n_features=1, noise=20, random_state=4
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234
)

# plot the training data vs the ouput data it looks like a linear model hmm.

# print(X.shape)
# print(y.shape)
# fig = plt.figure(figsize=(8,6))
# plt.scatter(X[:, 0],y,color="b",marker='o',s=30)
# plt.show()

def MSE(y_true,y_pred):
    return np.mean((y_true-y_pred)**2)

from linear_regression import LinearRegression

# create an instance of LinearRegression
alpha = [ 0.5,0.1,0.01]
for i in range(3):
        
    lr = LinearRegression(learning_rate = alpha[i])

    # fit the model
    lr.fit(X_train, y_train)

    # make predictions
    y_pred = lr.predict(X_test)

    # claculate the mean squared erro 
    mse_value  = MSE(y_test, y_pred)

    # print it 
    print(mse_value)


y_pred_line = lr.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8, 6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X, y_pred_line, color="black", linewidth=2, label="Prediction")
plt.show()
