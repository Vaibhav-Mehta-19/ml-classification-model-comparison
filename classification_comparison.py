#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
fruits= pd.read_table('fruits.txt')
print(fruits.head())


# In[2]:


print(fruits.shape)


# In[3]:


print(fruits['fruit_name'].unique())


# In[4]:


print(fruits.groupby('fruit_subtype').size())


# In[5]:


import seaborn as sb
sb.countplot(fruits['fruit_name'],label='Count')
plt.show()


# In[6]:


fruits.drop('fruit_label',axis=1).plot(kind='box',subplots=True,layout=(2,2),sharex=False, sharey=False, figsize=(9,9),title='Box plot for each one')
plt.show()


# In[7]:


import pylab as pl
fruits.drop('fruit_label',axis=1).hist(bins=50, figsize=(9,9))
pl.suptitle('Histogram')
plt.show()


# In[8]:


#train test and split
feature_names = ['mass','width','height','color_score']
X=fruits[feature_names]
y=fruits['fruit_label']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)
gamma='auto'


# In[10]:


#code preprocessing to bring values in same range
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


# In[11]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
model= LogisticRegression()
model.fit(X_train,y_train)
print('Train case accuracy is {:.2f}'.format(model.score(X_train,y_train)))
print('Test case accuracy is {:.2f}'.format(model.score(X_test,y_test)))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[17]:


#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
model= DecisionTreeClassifier()
model.fit(X_train,y_train)
print('Train case accuracy is {:.2f}'.format(model.score(X_train,y_train)))
print('Test case accuracy is {:.2f}'.format(model.score(X_test,y_test)))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[14]:


#KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
model= KNeighborsClassifier()
model.fit(X_train,y_train)
print('Train case accuracy is {:.2f}'.format(model.score(X_train,y_train)))
print('Test case accuracy is {:.2f}'.format(model.score(X_test,y_test)))
pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))                                                        


# In[15]:


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
model= GaussianNB()
model.fit(X_train,y_train)
print('Train case accuracy is {:.2f}'.format(model.score(X_train,y_train)))
print('Test case accuracy is {:.2f}'.format(model.score(X_test,y_test)))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[16]:


#SVM(SUPPORT VECTOR MACHINES)
from sklearn.svm import SVC
model= SVC()
model.fit(X_train,y_train)
print('Train case accuracy is {:.2f}'.format(model.score(X_train,y_train)))
print('Test case accuracy is {:.2f}'.format(model.score(X_test,y_test)))
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:





# In[ ]:




