from service.ModelService import load_model_as_np
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
import heatmap

utility, test, user, item, user_user_pearson = load_model_as_np()

# X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(utility, test, test_size=0.2, random_state=0)
print(" test")
