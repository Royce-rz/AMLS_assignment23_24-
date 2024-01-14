from A import Logistic_Regression
from A import SVM
from A import DecisionTree
from A import Data
from B import Logistic_Regression_B
from B import CNN

if __name__ == "__main__":
    print('The labels and images in Task A are :')
    Data.imageA()
    
    print('The Labels and images in Task B are ')
    Data.imageB()
    print('-----------------------------')
    print('##########################################')
   
    
    
    
    print('Task A by using Logistic Regresion')
    Logistic_Regression.LRans()
    print('-----------------------------')
    print('##########################################')



    print('Task A by using SVM')
    SVM.svmans()
    print('-----------------------------')
    print('##########################################')


    print('Task A by using Decision Tree')
    DecisionTree.DecisionTreeans()
    print('-----------------------------')
    print('##########################################')
    
    print('Task B by using Logistic Regresion')
    Logistic_Regression_B.LRtaskBans()
    print('-----------------------------')
    print('##########################################')
    
    print('CNN in Task B')
    CNN.CNNans()
