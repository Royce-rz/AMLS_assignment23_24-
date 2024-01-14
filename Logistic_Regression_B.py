#Importing necessary libraries for data manipulation, numerical computations, and plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing Logistic Regression model and performance evaluation metrics from sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

# Suppressing warnings to avoid unwanted output during model training and prediction
import warnings
warnings.filterwarnings('ignore')

# Loading the dataset from a specified file path
dataset_path = './Datasets/pathmnist.npz'
dataset = np.load(dataset_path)

# Reshaping training images for model input: flattening image data
x_train = dataset['train_images'].reshape([dataset['train_images'].shape[0], -1])
# Flattening training labels to a 1D array
y_train = dataset['train_labels'].ravel()

# Reshaping test images for model input: flattening image data
X_test = dataset['test_images'].reshape([dataset['test_images'].shape[0], -1])
# Flattening test labels to a 1D array
Y_test = dataset['test_labels'].ravel()

# Reshaping validation images for model input: flattening image data
X_val = dataset['val_images'].reshape([dataset['val_images'].shape[0], -1])
# Flattening validation labels to a 1D array
Y_val = dataset['val_labels'].ravel()

# Concatenating validation data with training data to increase training dataset size
X_train = np.concatenate((x_train, X_val), axis=0)
Y_train = np.concatenate((y_train, Y_val), axis=0)

# Defining a function for training and prediction using Logistic Regression
def logRegrPredict(x_train, y_train, xtest):
    # Building the Logistic Regression model
    logreg = LogisticRegression(solver='lbfgs')
    # Training the model with training data
    logreg.fit(x_train, y_train)
    # Predicting labels for the test dataset
    y_pred = logreg.predict(xtest)
    # Returning predictions and the trained model
    return y_pred, logreg

# Getting predictions and the trained model by calling the function with training and test data
y_pred, logreg = logRegrPredict(X_train, Y_train, X_test)
def LRtaskBans():
    # Printing the confusion matrix to evaluate model performance
    print(confusion_matrix(Y_test, y_pred))#plot
    # Printing the accuracy of the model on the test set
    print('Accuracy on test set:', accuracy_score(Y_test, y_pred))
    # Printing a detailed classification report
    print(classification_report(Y_test, y_pred))

if __name__ == '__main__':
    LRtaskBans()
