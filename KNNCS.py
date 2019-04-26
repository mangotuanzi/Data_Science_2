import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
class KNNClassifier:
    def __init__(self,k):
        assert k>=1,'k must be valid'
        self.k=k
        self._x_train=None
        self._y_train=None
 
    def fit(self,x_train,y_train):
        self._x_train=x_train
        self._y_train=y_train
        return self
 
    def _predict(self,x):
        d=[(-1*cosine_similarity([x,x_i])[0][1]) for x_i in self._x_train]
        nearest=np.argsort(d)
        top_k=[self._y_train[i] for i in nearest[:self.k]]
        votes=Counter(top_k)
        return votes.most_common(1)[0][0]
 
    def predict(self,X_predict):
        y_predict=[self._predict(x1) for x1 in X_predict]
        return np.array(y_predict)
 
    def __repr__(self):
        return 'knn(k=%d):'%self.k
 
    def score(self,x_test,y_test):
        y_predict=self.predict(x_test)
        return sum(y_predict==y_test)/len(x_test)
