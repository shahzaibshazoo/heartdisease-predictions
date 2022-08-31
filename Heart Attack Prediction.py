#!/usr/bin/env python
# coding: utf-8

# # Import Important Libraries

# In[98]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from featurewiz import featurewiz


# # Reading the dataSet

# In[99]:


df=pd.read_csv('heart.csv')
df.head()


# # Looking For null Values

# In[100]:


df.isnull().sum()


# # Identify Categorical Columns

# In[101]:


catCols = df.select_dtypes("object").columns
catCols


# # Convert Categorical to Integer Values

# In[102]:


le=LabelEncoder()
for i in catCols:
    df[i]=le.fit_transform(df[i])


# In[103]:


df


# # Feature Selection using Featurewix

# In[132]:


target = 'HeartDisease'
 
features, train = featurewiz(df, target, corr_limit=0.8)#, verbose=2, sep=",",header=0,test_data="", feature_engg="", category_encoders="")


# In[133]:


features


# In[134]:


X=df[features]
y=df[target]


# In[135]:


X.shape


# In[136]:


y.shape


# In[137]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)


# In[147]:


from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
model=SVC(kernel='linear')
model.fit(X_train,y_train)
model.score(X_test,y_test)


# In[149]:


import joblib
joblib.dump(model,'HeartAttack.pkl')


# In[ ]:




