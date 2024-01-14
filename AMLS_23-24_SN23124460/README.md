# Medical Image Analysis Project

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
1. Set the `task` variable in `main.py` to either "A" for PneumoniaMNIST or "B" for PathMNIST.
2. Set the `model` variable to "SVM", "CNN", or "Random Forest" depending on the model you wish to use.
3. Optionally, set the `retrain` flag to `True` if you want to retrain the model, or `False` to use a pre-trained model.

## Requirements

To run this project, your environment need to include Python 3.8 and several packages. The `environment.yml` file in this repository lists all the necessary dependencies.
The project requires the following packages:
- numpy
- torch
- joblib
- medmnist
- sklearn
- matplotlib

## Notes
- Ensure all the paths and file names are correct and match your project directory structure.
- Adjust the README as needed to reflect any changes or additions to your project.
