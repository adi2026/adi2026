#!/usr/bin/env python
# coding: utf-8

# In[92]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


# In[93]:


titanic = pd.read_csv("tested.csv")
titanic.head()


# In[94]:


titanic.shape


# In[95]:


sns.countplot(x='Survived', data=titanic)


# In[96]:


sns.countplot(x='Survived', hue='Sex', data=titanic, palette='winter')


# In[97]:


sns.countplot(x='Survived', hue='Pclass',data=titanic,palette='PuBu')


# In[98]:


titanic['Age'].plot.hist()


# In[15]:


titanic['Fare'].plot.hist(bins=20, figsize=(10,5))


# In[16]:


sns.countplot(x='SibSp', data=titanic,palette='rocket')


# In[17]:


sns.countplot(x='Survived', data=titanic)


# In[18]:


sns.countplot(x='Survived', hue='Sex', data=titanic , palette='winter')


# In[19]:


sns.countplot(x='Survived', hue='Pclass', data=titanic, palette='PuBu')


# In[20]:


titanic['Age'].plot.hist()


# In[22]:


titanic['Fare'].plot.hist(bins=20,figsize=(10,5))


# In[23]:


sns.countplot(x='SibSp', data=titanic,palette='rocket')


# In[24]:


titanic['Parch'].plot.hist()


# In[25]:


sns.countplot(x='Parch', data=titanic,palette='summer')


# In[101]:


titanic.isnull()


# In[32]:


sns.heatmap(titanic.isnull(),cmap='spring')


# In[38]:


plt.figure(figsize=(8,8))
sns.boxplot(x="Pclass", y="Age", data=titanic)
plt.show()


# In[39]:


titanic.head()


# In[42]:


titanic.drop('Cabin',axis=1, inplace=True)


# In[43]:


titanic.head(3)


# In[44]:


titanic.dropna(inplace=True)


# In[47]:


sns.heatmap(titanic.isnull(), cbar=False)


# In[48]:


titanic.isnull().sum()


# In[50]:


titanic.head(2)


# In[52]:


pd.get_dummies(titanic['Sex']).head(5)


# In[54]:


sex=pd.get_dummies(titanic['Sex'],drop_first=True)
sex.head(3)


# In[55]:


embark=pd.get_dummies(titanic['Embarked'])


# In[56]:


embark.head(3)


# In[57]:


embark=pd.get_dummies(titanic['Embarked'],drop_first=True)


# In[58]:


embark.head(3)


# In[62]:


pcl=pd.get_dummies(titanic['Pclass'],drop_first=True)
pcl.head(3)


# In[63]:


tianic=pd.concat([titanic, sex, embark,pcl],axis=1)


# In[64]:


titanic.head(3)


# In[72]:


titanic.drop(['Name','PassengerId','Pclass',"Ticket",'Sex','Embarked'], axis=1, inplace=True)


# In[73]:


titanic.head(3)


# In[74]:


x=titanic.drop('Survived', axis=1)
y=titanic['Survived']


# In[76]:


from sklearn.model_selection import train_test_split


# In[77]:


x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.33, random_state=4)


# In[78]:


from sklearn.linear_model import LogisticRegression
lm=LogisticRegression()


# In[79]:


lm.fit(x_train,y_train)


# In[81]:


prediction=lm.predict(x_test)


# In[82]:


from sklearn.metrics import classification_report


# In[83]:


classification_report(y_test, prediction)


# In[86]:


from sklearn.metrics import confusion_matrix


# In[87]:


confusion_matrix(y_test, prediction)


# In[88]:


from sklearn.metrics import accuracy_score


# In[89]:


accuracy_score(y_test,prediction)


# In[ ]:




