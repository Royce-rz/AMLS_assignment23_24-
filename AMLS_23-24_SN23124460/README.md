# AMLS_Assignment 23-24

## Project Description
This project is to use different algorithm to classify the two different datasets which are PneumoniaMNIST dataset and PathMNIST dataset
## File Structure
- `main.py`: The main executable script for the project. It handles data loading, model selection, training, and testing.
- `A/`:
  - `Logistic_Regression.py`: Use Logistic Regression for binary classification on the PneumoniaMNIST dataset.
  - `DecisionTree.py`: Use Decision Tree on the PneumoniaMNIST dataset.
  - `SVM.py`: Use SVM on the PneumoniaMNIST dataset.
- `B/`:
  - `Logistic_RegressionB.py`: Use Logistic Regression for multi-class classificationon the PathMNIST dataset.
  - `CNN.py`: Contains functions for training and testing the CNN model on the PathMNIST dataset.
- `Datasets/`: Directory where the PneumoniaMNIST and PathMNIST datasets are stored.

## How to Run
1. Upload the Datasets to `Datasets` folder, the name should be pneumoniamnist.npz for task A and pathmnist.npz for task B
2. Get all the environment requirement and go to 'main.py'
## Requirements

To run this project, your environment need to include Python 3.11.5 and several packages. 
The project requires the following packages:
- numpy
- warning
- matplotlib.pyplot
- matplotlib
- tensorflow
- sklearn


