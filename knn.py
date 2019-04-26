from sklearn import datasets#导入数据
from sklearn.model_selection import train_test_split
from sklearn.neighbors import  KNeighborsClassifier
from sklearn import metrics
import numpy as np
y=np.load('label.npy')


x=np.load('feature.npy')
   
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
 
knn = KNeighborsClassifier(n_neighbors=7, metric='euclidean')

knn.fit(x_train,y_train)
y_pre = knn.predict(x_test)
print("Euclidean")
print(metrics.accuracy_score(y_pre,y_test))

    
knn = KNeighborsClassifier(n_neighbors=7, metric='manhattan')

knn.fit(x_train,y_train)
y_pre = knn.predict(x_test)
print("Manhattan")
print(metrics.accuracy_score(y_pre,y_test))


knn = KNeighborsClassifier(n_neighbors=7, metric='chebyshev')

knn.fit(x_train,y_train)
y_pre = knn.predict(x_test)
print("Chebyshev")
print(metrics.accuracy_score(y_pre,y_test))


