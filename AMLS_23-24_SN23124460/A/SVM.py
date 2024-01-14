
# import libraries
from sklearn.model_selection import StratifiedKFold
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = np.load('./Datasets/pneumoniamnist.npz')

# Extracting the data
train_images = dataset['train_images']
val_images = dataset['val_images']
test_images = dataset['test_images']
y_train = train_labels = dataset['train_labels'].ravel()
y_val = val_labels = dataset['val_labels'].ravel()
y_test = test_labels = dataset['test_labels'].ravel()

# Reshape and normalize the images
X_train = train_images.reshape(train_images.shape[0], -1) / 255.0
X_val = val_images.reshape(val_images.shape[0], -1) / 255.0
X_test = test_images.reshape(test_images.shape[0], -1) / 255.0

#Train a basic SVM model
svm_model = SVC(class_weight='balanced')
svm_model.fit(X_train, y_train)

# Evaluate the basic model on the test set
y_pred_test = svm_model.predict(X_test)
basic_accuracy = accuracy_score(y_test, y_pred_test)
basic_classification_report = classification_report(y_test, y_pred_test, zero_division=0)

# print("Basic Model Accuracy:", basic_accuracy)
# print("Classification Report:\n", basic_classification_report)

# create a funtion to evaluate the model
def evaluate_svm(C, gamma, class_weight, X_train, y_train, X_val, y_val):
    model = SVC(C=C, gamma=gamma, kernel='rbf', class_weight=class_weight, random_state=42)
    model.fit(X_train, y_train)
    y_pred_val = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred_val)
    classification_rep = classification_report(y_val, y_pred_val)
    return accuracy, classification_rep

# Manually testing different hyperparameters
hyperparameters = [
    (0.1, 'scale', None),
    (1, 'scale', None),
    (10, 'scale', None),
    (0.1, 'auto', None),
    (1, 'auto', None),
    (10, 'auto', None),
    (0.1, 'scale', 'balanced'),
    (1, 'scale', 'balanced'),
    (10, 'scale', 'balanced'),
    (0.1, 'auto', 'balanced'),
    (1, 'auto', 'balanced'),
    (10, 'auto', 'balanced')
]

# Initialize variables to store the best parameters and the highest accuracy
best_accuracy = 0
best_params = {'C': None, 'gamma': None, 'class_weight': None}

# Evaluate each combination on the validation set
for C, gamma, class_weight in hyperparameters:
    accuracy, report = evaluate_svm(C, gamma, class_weight, X_train, y_train, X_val, y_val)
    # print(f"Parameters: C={C}, gamma={gamma}, class_weight={class_weight}")
    # print(f"Validation Accuracy: {accuracy}")
    # print(f"Validation Classification Report:\n{report}\n")
    
     # Update the best parameters if current accuracy is higher
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_params = {'C': C, 'gamma': gamma, 'class_weight': class_weight}

def svmans():      
    print(f"best_params = {best_params}")        
    # Use the best parameters to train the final model and evaluate on the test set
    final_model = SVC(**best_params, random_state=30)
    final_model.fit(X_train, y_train)  
    y_pred_test = final_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    print("Model Accuracy:", basic_accuracy)
    print("Classification Report:\n", basic_classification_report)
    print(f"\nTest Accuracy with Best Parameters: {test_accuracy}")
    print(f"Test Classification Report:\n{test_report}")

if __name__ == "__main__":
    svmans()