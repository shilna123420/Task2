#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from  nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


# pip install scikit-learn==0.13


# In[32]:


import nltk 
# nltk.download('stopwords')


# In[3]:


print(stopwords.words('english'))


# In[4]:


df=pd.read_csv('twitter.csv',encoding='ISO-8859-1')
df


# In[7]:


df.head()


# In[8]:


col_names=['target','id','date','flag','user','text']
df.columns=col_names
df.head()


# In[9]:


df.shape


# In[10]:


df.isnull().sum()


# In[11]:


df['target'].value_counts()


# In[12]:


df['target']=df['target'].map({4:1,0:0})


# In[13]:


#stemming

stremmer=PorterStemmer()

def stemming(content):
    stemmed_content=re.sub('[^a-zA-z]',' ',content) #removing not a-z and A-Z
    stemmed_content=stemmed_content.lower()
    stemmed_content=stemmed_content.split()
    stemmed_content=[stremmer.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content=' '.join(stemmed_content)
    return stemmed_content


# In[ ]:


df['text']=df['text'].apply(stemming)


# In[ ]:


# df.head()


# In[ ]:


df.head()


# In[ ]:


df


# In[23]:


x=df['text']
y=df['target']


# In[24]:


#splitting data set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[26]:


#convert textual data into numerical_data

vectorizer=TfidfVectorizer()
x_train=vectorizer.fit_transform(x_train)
x_test=vectorizer.transform(x_test)


# In[30]:


print(x_train)


# In[31]:


#training model

model=LogisticRegression()
model.fit(x_train,y_train)


# In[33]:


#testing model

y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))


# In[34]:


#function to predict the sentiment

def predict_sentiment(text):
    text=re.sub('[^a-zA-Z]',' ',text)
    text=text.lower()
    text=text.split()
    text=[stremmer.stem(word) for word in text if not word in stopwords.words('english')]
    text=' ' .join(text)
    text=[text]
    text=vectorizer.transform(text)
    sentiment=model.predict(text)
    if sentiment==0:
        return "Negative"
    else:
        return "Positive"


    


# In[35]:


#testing the model

print(predict_sentiment("i hate you"))
print(predict_sentiment("i love you"))


# In[ ]:





# In[ ]:





# In[ ]:




