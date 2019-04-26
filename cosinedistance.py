import numpy as np
from sklearn.model_selection import train_test_split
from KNNCS import KNNClassifier
from sklearn.metrics import accuracy_score
y=np.load('label.npy')
x=np.load('feature.npy')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
my_knn=KNNClassifier(k=7)
my_knn.fit(x_train,y_train)
y_predict=my_knn.predict(x_test)
print(accuracy_score(y_test,y_predict))
score=my_knn.score(x_test,y_test)
print(score)
