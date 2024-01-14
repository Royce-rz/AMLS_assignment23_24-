#Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree# Load the dataset

dataset_path = './Datasets/pneumoniamnist.npz'
dataset = np.load(dataset_path)

# Reshaping the data
X_train = dataset['train_images'].reshape([dataset['train_images'].shape[0], -1])
Y_train = dataset['train_labels'].ravel()  # Flattening to 1D array
X_test = dataset['test_images'].reshape([dataset['test_images'].shape[0], -1])
Y_test = dataset['test_labels'].ravel()  # Flattening to 1D array


#print('Number of examples in the data:', X_train.shape[0])




#Importing the Decision tree classifier from the sklearn library.
tree_params={
    'criterion':'entropy',
    'min_samples_split':10
}
clf = tree.DecisionTreeClassifier( **tree_params )

#Training the decision tree classifier on training set. 
# Please complete the code below.
clf.fit(X_train,Y_train)


#Predicting labels on the test set.
# Please complete the code below.
y_pred =  clf.predict(X_test)
def DecisionTreeans():
    #print(f'Test feature {X_test[510]}\n True class {Y_test[510]}\n predict class {y_pred[510]}')
    #Use accuracy metric from sklearn.metrics library
    print('Accuracy Score on train data: ', accuracy_score(y_true=Y_train, y_pred=clf.predict(X_train)))
    print('Accuracy Score on test data: ', accuracy_score(y_true=Y_test, y_pred=y_pred))


    # Setting the size of the plot
    plt.figure(figsize=(20,10))

    # Plotting the tree structure
    plot_tree(clf, 
            filled=True, 
            rounded=True, 
            class_names=['Class 0', 'Class 1'], 
            feature_names=[f'feature_{i}' for i in range(X_train.shape[1])])

    # Showing the plot
    plt.show()
    
if __name__ == "__main__":
    DecisionTreeans()
