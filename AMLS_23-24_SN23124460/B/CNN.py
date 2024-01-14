import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import optimizers, losses
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers.legacy import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.preprocessing import label_binarize

# Load the dataset
data = np.load('./Datasets/pathmnist.npz')

x_train = data['train_images']
y_train = data['train_labels']
x_test = data['test_images']
y_test = data['test_labels']
x_val = data['val_images']
y_val = data['val_labels'].ravel()

# Normalize the data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_val = x_val.astype('float32') / 255.0


# One-hot encode the labels
num_classes = len(np.unique(y_train))
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

# Build the model
new_model = tf.keras.Sequential([
    Conv2D(16, 3, padding="same", input_shape=(28, 28, 3), activation="tanh"),
    MaxPool2D((2,2)),
    Conv2D(32, 3, padding="same", activation="tanh"),
    MaxPool2D((3,3)),
    Conv2D(64, 3, padding="same", activation="tanh"),
    MaxPool2D((3,3)),
    Flatten(),
    Dense(200, activation="tanh"),
    Dense(50, activation="tanh"),
    Dense(num_classes, activation="softmax")  # Adjusted to match the number of classes
])

new_model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss=losses.categorical_crossentropy,
                  metrics=["accuracy"])

# # Train the model
# history = new_model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=32, epochs=20)
from tensorflow.keras.callbacks import EarlyStopping

# Define early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

# Train the model with early stopping
history = new_model.fit(x_train, y_train, validation_data=(x_val, y_val), 
                        batch_size=16, epochs=50, callbacks=[early_stopping])
def CNNans():
    # Plot the training and validation loss
    plt.plot(history.history["loss"], label="train_set")
    plt.plot(history.history["val_loss"], label="test_set")
    plt.title('Model Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()




    # Plot the training and validation accuracy
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()


    # Make predictions
    y_pred = new_model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)

    # Calculate F1-score, sensitivity, and specificity
    print(classification_report(y_true_classes, y_pred_classes))

    # Binarize the labels for AUC-ROC
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_true_classes))

    # Compute ROC curve and ROC area for each class
    n_classes = num_classes
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_pred.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    lw = 2
    plt.plot(fpr["micro"], tpr["micro"],
            label='Micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    CNNans()