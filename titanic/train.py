# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df.head()

data_test = combine[0]
Y_test = data_test['Survived']
X_test = data_test.drop('Survived', axis=1)

SVC= SVC()
SVC.fit(X_train, Y_train)
Y_pred = SVC.predict(X_test)
acc_SVC = round(SVC.score(X_train, Y_train) * 100, 2)
acc_SVC
