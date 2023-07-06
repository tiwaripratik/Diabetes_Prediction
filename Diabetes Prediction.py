#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[2]:


df=pd.read_csv("D:\ML\dataset\diabetes.csv")


# In[3]:


df


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


df.describe()


# In[8]:


df['Outcome'].value_counts()


# In[9]:


df.groupby('Outcome').mean()


# In[10]:


X=df.drop(columns='Outcome',axis=1)
Y=df['Outcome']


# In[11]:


X


# In[12]:


Y


# In[13]:


scaler=StandardScaler()


# In[14]:


scaler.fit(X)


# In[15]:


standardized=scaler.transform(X)


# In[16]:


print(standardized)


# In[17]:


X=standardized


# In[18]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)


# In[19]:


print(X.shape,X_train.shape,X_test.shape,)


# In[20]:


classifier=svm.SVC(kernel='linear')


# In[21]:


classifier.fit(X_train,Y_train)


# In[22]:


X_train_prediction=classifier.predict(X_train)
training_accuracy=accuracy_score(X_train_prediction,Y_train)


# In[23]:


print("The accuracy score of the  training data:",training_accuracy)


# In[24]:


X_test_prediction=classifier.predict(X_test)
testing_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[25]:


print("The accuracy score of the  testing data:",testing_accuracy)


# In[26]:


input_data=(4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction=classifier.predict(std_data)
print(prediction)
if (prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")


# In[27]:


import pickle


# In[28]:


filename='diabetic.sav'


# In[30]:


pickle.dump(classifier,open(filename,'wb'))


# In[32]:


loaded_model=pickle.load(open('diabetic.sav','rb'))


# In[33]:


input_data=(4,110,92,0,0,37.6,0.191,30)
input_data_as_numpy=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy.reshape(1,-1)
std_data=scaler.transform(input_data_reshaped)
print(std_data)
prediction=loaded_model.predict(std_data)
print(prediction)
if (prediction[0]==0):
    print("The person is not diabetic")
else:
    print("The person is diabetic")


# In[ ]:




