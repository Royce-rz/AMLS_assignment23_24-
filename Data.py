
import numpy as np
import matplotlib.pyplot as plt

def imageA():
    # Load the dataset
    data = np.load('./Datasets/pneumoniamnist.npz')

    # Extract the training images and labels
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    # Display the shape of the data
    print("PneumoniaMNIST Dataset")
    print("Training Images Shape:", train_images.shape)
    print("Training Labels Shape:", train_labels.shape)
    print('Validation Labels Shape:', val_labels.shape)
    print('Validation Images Shape:', val_images.shape)
    print('Test Labels Shape:', test_labels.shape)
    print('Test Images Shape:', test_images.shape)

    # Find the indices of the first occurrences of the labels 1 and 0
    index_of_1 = np.where(train_labels == 1)[0][0]
    index_of_0 = np.where(train_labels == 0)[0][0]

    # Plotting the images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(train_images[index_of_1].squeeze(), cmap='gray')
    plt.title('Pneumonia (Label 1)')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(train_images[index_of_0].squeeze(), cmap='gray')
    plt.title('Normal (Label 0)')
    plt.axis('off')

    plt.show()
    
    
def imageB():
    data = np.load('./Datasets/pathmnist.npz')
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']
    # Display the shape of the data
    print("PathMNIST Dataset")
    print("Training Images Shape:", train_images.shape)
    print("Training Labels Shape:", train_labels.shape)
    print('Validation Labels Shape:', val_labels.shape)
    print('Validation Images Shape:', val_images.shape)
    print('Test Labels Shape:', test_labels.shape)
    print('Test Images Shape:', test_images.shape)

    # Plotting one image from each class
    num_classes = 9
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for i in range(num_classes):
        idx = np.where(train_labels == i)[0][0]
        img = train_images[idx]

        row, col = divmod(i, 3)
        ax = axes[row, col]
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Class {i}')

    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    imageA()
    imageB()
