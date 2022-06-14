# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values

## Program:
#Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#Developed by: bhuvaneshwari.s
#RegisterNumber: 212221240010
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

## Output:
Head:
![1](https://user-images.githubusercontent.com/94828604/173535730-0277f499-d426-4514-9547-b843ebe22b3a.png)


Predicted Values:

![2](https://user-images.githubusercontent.com/94828604/173535819-ff26d504-8414-4061-abc7-392db07dd714.png)

Accuracy:
![3](https://user-images.githubusercontent.com/94828604/173535937-c5a96495-9e4b-428d-99f7-ab6f73e7ade2.png)


Confusion Matrix:
![4](https://user-images.githubusercontent.com/94828604/173536021-11b68f0b-7517-4789-b397-b9aa510326e9.png)


Classification Report:

![5](https://user-images.githubusercontent.com/94828604/173536094-2382ed95-ba4a-455d-8509-ee159ff77d55.png)


## Result:
Thus, the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
