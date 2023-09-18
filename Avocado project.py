#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt


# In[13]:


df_tr=pd.read_csv(r"C:\Users\HP\Downloads\avocado.csv.zip")
df_tr


# In[14]:


df_tr.shape


# In[15]:


df_tr.isnull()


# In[16]:


df_tr.isnull().sum()


# In[17]:


sns.heatmap(df_tr.isnull())
plt.show()


# In[18]:


df_tr.isna().sum()


# In[19]:


df_tr.columns


# In[20]:


df_tr.info()


# In[21]:


df_tr.dtypes


# In[23]:


df_tr.drop(['Date'],axis=1,inplace=True)
df_tr.head()


# In[24]:


df_tr.columns


# In[25]:


df_tr['AveragePrice'].unique()


# In[26]:


df_tr['Total Volume'].unique()


# In[29]:


df_tr['4046'].unique()


# In[30]:


df_tr['4225'].unique()


# In[31]:


df_tr['4770'].unique()


# In[32]:


df_tr['Total Bags'].unique()


# In[33]:


df_tr['Small Bags'].unique()


# In[35]:


df_tr['XLarge Bags'].unique()


# In[37]:


df_tr['type'].unique()


# In[38]:


df_tr['year'].unique()


# In[39]:


df_tr['region'].unique()


# In[41]:


sns.histplot(df_tr['AveragePrice'])
plt.show()


# In[42]:


sns.histplot(df_tr['Total Volume'])
plt.show()


# In[43]:


sns.histplot(df_tr['4046'])
plt.show()


# In[44]:


sns.histplot(df_tr['4225'])
plt.show()


# In[46]:


sns.histplot(df_tr['4770'])
plt.show()


# In[47]:


sns.histplot(df_tr['Total Bags'])
plt.show()


# In[49]:


sns.histplot(df_tr['Small Bags'])
plt.show()


# In[50]:


sns.histplot(df_tr['Large Bags'])
plt.show()


# In[51]:


sns.histplot(df_tr['XLarge Bags'])
plt.show()


# In[52]:


sns.histplot(df_tr['type'])
plt.show()


# In[53]:


sns.histplot(df_tr['year'])
plt.show()


# In[54]:


sns.histplot(df_tr['region'])
plt.show()


# In[56]:


from sklearn.preprocessing import LabelEncoder # import


# In[57]:


le=LabelEncoder()
for i in df_tr.drop(['AveragePrice'],axis=1):
    df_tr[i]=le.fit_transform(df_tr[i])
df_tr


# In[58]:


le=LabelEncoder()
for i in df_tr.drop(['Total Volume'],axis=1):
    df_tr[i]=le.fit_transform(df_tr[i])
df_tr


# In[59]:


df_tr.head()


# In[60]:


df_tr.dtypes


# In[61]:


# plot graph for co-relation in Bi Variate Analysis
import seaborn as sns
for col in df_tr.drop(['AveragePrice'],axis=1):
    plt.figure(figsize=(6,4))
    plt.title(f'{col} vs. AveragePrice')
    sns.scatterplot(y=df_tr[col],x=df_tr['AveragePrice'],hue=df_tr['AveragePrice'])
    plt.show()


# In[62]:


df_tr.corr()


# In[63]:


plt.figure(figsize=(15,8))
sns.heatmap(df_tr.corr(),annot=True)


# In[ ]:




