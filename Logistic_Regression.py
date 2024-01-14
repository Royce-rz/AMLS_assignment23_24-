
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from scipy.special import expit as sigmoid
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
# Suppressing warnings to avoid unwanted output during model training and prediction
import warnings
warnings.filterwarnings('ignore')




def sigmoid(z):
    return 1. / (1. + np.exp(-z))

# Logistic Regression Parameter Estimation Function
def logRegParamEstimates(X, y):
    intercept = np.ones((X.shape[0], 1))
    X = np.concatenate((intercept, X), axis=1)
    theta = np.zeros(X.shape[1])
    for i in range(100):
        z = np.dot(X, theta)
        h = sigmoid(z)
        lr = 0.04
        gradient = np.dot(X.T, (h - y)) / y.shape[0]
        theta -= lr * gradient
    return theta
    # Logistic Regression Prediction Function
def logRegrNEWRegrPredict(X_train, y_train, X_test):
    theta = logRegParamEstimates(X_train, y_train)
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1)
    sig = sigmoid(np.dot(X_test, theta))
    return sig >= 0.5 # Return True or False predictions

# Modified prediction function to return probabilities
def logRegrNEWRegrPredict_proba(X_train, y_train, X_test):
    theta = logRegParamEstimates(X_train, y_train)
    intercept = np.ones((X_test.shape[0], 1))
    X_test = np.concatenate((intercept, X_test), axis=1)
    return sigmoid(np.dot(X_test, theta))  # Return probabilities

def LRans():
    
    # Load the dataset
    dataset_path = './Datasets/pneumoniamnist.npz'
    dataset = np.load(dataset_path)

    # Reshape and flatten the training and test data
    X_train = dataset['train_images'].reshape([dataset['train_images'].shape[0], -1])
    y_train = dataset['train_labels'].ravel()
    X_test = dataset['test_images'].reshape([dataset['test_images'].shape[0], -1])
    y_test = dataset['test_labels'].ravel()

    # Ensure binary classification: Convert labels greater than 1 to 1
    y_train = np.where(y_train > 1, 1, y_train)
    y_test = np.where(y_test > 1, 1, y_test)






    # Predict using the logistic regression model
    y_pred = logRegrNEWRegrPredict(X_train, y_train, X_test)

    # Evaluate the model

    print('Confusion matrix taskA LR',confusion_matrix(y_test, y_pred))#plot
    # Printing the accuracy of the model on the test set
    print('Accuracy on Task A LR test set:', accuracy_score(y_test, y_pred))
    # Printing a detailed classification report
    print(classification_report(y_test, y_pred))



    # Get predicted probabilities
    y_pred_proba = logRegrNEWRegrPredict_proba(X_train, y_train, X_test)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    # Plotting the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()



if __name__ == "__main__":
    LRans()