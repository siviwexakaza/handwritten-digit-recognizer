#Digit Recognizer Machine Learning Script
#Author: Siviwe Xakaza, siviwexakaza@gmail.com

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
#importing the dataset
data = pd.read_csv("train.csv").as_matrix()
#creating an instance of DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
#Selecting the 1st 21000 rows from the dataset but excluding the first column
train_data=data[0:21000,1:]
#Selecting the 1st 21000 rows from the dataset but only the first column, this is what the model will predict
training_label = data[0:21000,0]
#Training the model
decision_tree.fit(train_data,training_label)
#Selecting the remaining rows for both the label and features
test_data=data[21000:,1:]
testing_label=data[21000:,0]
#picking a random digit
digit = test_data[15]
digit.shape=(28,28)
plt.imshow(255-digit,cmap='gray')
print(decision_tree.predict([test_data[15]]))
plt.show()
