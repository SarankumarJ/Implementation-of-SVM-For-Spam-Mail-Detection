# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

  1. Import the required packages.
  2. Import the dataset to operate on.
  3. Split the dataset.
  4. Predict the required output.
  5. End the program.

## Program:
```py
Program to implement the SVM For Spam Mail Detection..
Developed by: SARANKUMAR J
RegisterNumber:  212221230087

import pandas as pd
data=pd.read_csv("spam.csv",encoding='Windows-1252')

import chardet
file='spam.csv'
with open(file, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(10000))
result

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://user-images.githubusercontent.com/94778101/204443692-929e0881-bf2f-49f4-acb5-b1b7e865d6d7.png)

![image](https://user-images.githubusercontent.com/94778101/204443735-4c82f4c7-5d50-4260-8f47-0d680a1d6add.png)





![image](https://user-images.githubusercontent.com/94778101/204443756-a189a765-1622-4bb6-9d3d-94c42a942a72.png)

![image](https://user-images.githubusercontent.com/94778101/204443794-ebbbaf38-dbb9-42d5-a9ec-33744f72291a.png)

![image](https://user-images.githubusercontent.com/94778101/204443825-fd25b5ef-1a55-475f-8033-b9c6339c13a8.png)

![image](https://user-images.githubusercontent.com/94778101/204443845-b4748fd1-8c6e-44d5-b8f1-58cb15fd8380.png)

![image](https://user-images.githubusercontent.com/94778101/204443862-e78966a1-bc8a-4ca9-8d6d-79f691f9f7f5.png)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
