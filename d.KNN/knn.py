import numpy as np
from collections import Counter
# global function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k


    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x):
        #compute distance
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]

        # get k nearest samples ,labels
        k_indices = np.argsort(distances)[:self.k] # sort the indices and return indices from 0 to the k indice
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label
        most_common_class = Counter(k_nearest_labels).most_common(1) #return a list of tuple of the most common number that repeats with theire number of repetitions
        return most_common_class[0][0]


if __name__ == "__main__":
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    from matplotlib.colors import ListedColormap

    import matplotlib.pyplot as plt
    cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

    iris = datasets.load_iris()
    X,y = iris.data,iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )
    def info():
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)
        print(X_train[0])
        print(y_train[0])
    # info()
        
    def plot():
        plt.figure()
        plt.scatter(X[:,0],X[:,1],c=y,cmap = cmap,edgecolors='k',s=20)
        plt.show()

    # plot()
    clf = KNN(k=10)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)


    def accuracy(predictions):
        return np.sum(predictions == y_test)/len(y_test)
    
    accuracy = accuracy(y_pred)
    print(accuracy)

    







    