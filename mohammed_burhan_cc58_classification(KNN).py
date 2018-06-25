# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 10:49:02 2018

@author: Lenovo
"""

import pandas as pd
data=pd.read_csv("mushrooms.csv")


features=data.iloc[:,[5,21,22]].values

labels=data["class"].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()

for i in [0,1,2]:
   features[:,i]=le.fit_transform(features[:,i])
        
feature_df=pd.DataFrame(features)






#onehotencoding
ohe=OneHotEncoder(categorical_features=[0])
features=ohe.fit_transform(features).toarray()
#dummy variable trap
features=features[:,1:]

ohe=OneHotEncoder(categorical_features=[-2])
features=ohe.fit_transform(features).toarray()
#dummy variable trap
features=features[:,1:]

ohe=OneHotEncoder(categorical_features=[-1])
features=ohe.fit_transform(features).toarray()

#dummy variable trap
features=features[:,1:]



from sklearn.model_selection import train_test_split
features_train,features_test,labels_train,labels_test=train_test_split(features,labels,test_size=0.25,random_state=0)


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,p=2)
knn.fit(features_train,labels_train)

#prediction
labels_predict=knn.predict(features_test)

#score
score=knn.score(features_test,labels_test)

#making confussion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(labels_test,labels_predict)

#results prediction
labels_predict=knn.predict([])
