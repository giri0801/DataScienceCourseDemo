#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[47]:


df = pd.read_csv('happiness_score_dataset.csv');


# In[48]:


df.head()


# In[49]:


df.head(12)


# In[50]:


df.describe()


# In[51]:


df.shape


# In[52]:


df.isna().sum()


# In[53]:


import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# In[54]:


lab_Enc = LabelEncoder()
df2 = lab_Enc.fit_transform(df['Country'])
df['Country'] = df2
df3 = lab_Enc.fit_transform(df['Country'])
df['Region'] = df3


# In[55]:


df.head()


# In[56]:


df.skew()


# In[57]:


import matplotlib.pyplot as plt
plt.figure(figsize=(15,15))
plotnumber=1
for column in df:
    if(plotnumber <=12):
        plt.subplot(3,3,plotnumber)
        sns.distplot(df[column])
    plotnumber +=1  
plt.show()


# In[58]:


df.columns


# In[59]:


df.skew()


# In[60]:


# sqrt(x)
df['Standard Error']= np.log(df["Standard Error"])


# In[61]:


df['Trust (Government Corruption)']= np.sqrt(df["Trust (Government Corruption)"])


# In[62]:


df.skew()


# In[63]:


# df['Generosity']= np.sqrt(df["Generosity"])


# In[64]:


# outliers
df['Standard Error'].plot.box()


# In[65]:


dfCor = df.corr()
# df = df.drop(['Happiness Score'], axis=1)


# In[66]:


collist = df.columns
noOfCol = collist.values
# df = df.drop(['Region'], axis=1)


# In[67]:


dfCor = df.corr()


# In[68]:


plt.figure(figsize =(15,20))
sns.heatmap(dfCor,annot = True)
plt.show()


# In[69]:


df.corr()['Happiness Score'].sort_values()


# In[70]:


df = df.drop(['Happiness Rank','Standard Error','Region'],axis =1)


# In[71]:


plt.figure(figsize =(20,25))
# df = df.drop(['Country'], axis=1)

graph = 1
for col in df:
    if graph <=12:
        plt.subplot(3,3,graph)
        ax=sns.boxplot(data=df[col])
        plt.xlabel(col)
        graph+=1
plt.show        
        
        


# In[72]:


from scipy.stats import zscore


# In[73]:


zscore(df)


# In[74]:



z_score = np.abs(zscore(df))
z_score 


# In[75]:


np.where(z_score>3)


# In[76]:


df_new = df[(z_score < 3).all(axis=1)]


# In[77]:


df_new.shape


# In[78]:


# y = df_new['Happiness Score']
df.shape


# In[79]:


x = df_new.drop('Happiness Score', axis=1)
y = df_new['Happiness Score']


# In[ ]:





# In[80]:


from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar_data = scalar.fit_transform(x)


# In[81]:


from sklearn.model_selection import train_test_split


# In[82]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
x_train,x_test,y_train,y_test = train_test_split(scalar_data,y,test_size=0.33, random_state=42)


# In[83]:


lr.fit(x_train,y_train)


# In[84]:


lr.predict(scalar_data)


# In[ ]:





# In[85]:


lr.score(x_test,y_test)


# In[86]:


from sklearn.metrics import r2_score


# In[87]:


pred_train = lr.predict(x_train)
pred_test = lr.predict(x_test)
print("Accuracy score")
print(r2_score(y_train,pred_train))
print(r2_score(y_test,pred_test))


# In[ ]:





# In[ ]:





# In[ ]:




