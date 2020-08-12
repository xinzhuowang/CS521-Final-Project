#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import  train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
import numpy as np


# In[2]:


get_ipython().run_line_magic('cd', 'C:\\Users\\mac\\Desktop')
data = pd.read_csv("House Prediction Data.csv")
data.head(10)


# In[3]:


label = data["SalePrice"]
# features = data[['MSSubClass', 'LotArea', "OverallQual", "OverallCond", "MasVnrArea",
#                  "BsmtFinSF1","BsmtUnfSF" ,"TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea"]]
features = data[['YrSold','LotArea']]


# In[4]:


plt.figure()
plt.scatter(data["Id"], label, color = "blue")
plt.xlabel("Id")
plt.ylabel("SalePrice")


# In[5]:


plt.figure()
plt.bar(data["YrSold"], label,  color = "pink")
plt.xlabel("YrSold")
plt.ylabel("SalePrice")


# In[6]:


plt.figure()
plt.scatter(data["LotArea"], label,  color = "orange")
plt.xlabel("LotArea")
plt.ylabel("SalePrice")


# In[7]:


print("correlations of 'YrSold','LotArea' is ", np.corrcoef(data['YrSold'],data['LotArea'] ))
print("correlations of 'YrSold','OverallQual' is ", np.corrcoef(data['YrSold'],data['OverallQual'] ))


# In[8]:


print("features = ", features)
for i in range(len(label)):
    if pd.isnull(label.at[i])  :
        label = label.drop(index = i)
        features = features.drop(index = i)
        # print("true")
        pass
print("data = ", data)
print("label = ", label)


# In[9]:


features = preprocessing.scale(features)
X_train,X_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=0)


# In[10]:


linear = LinearRegression()
linear.fit(X_train, y_train)
# print("Liner Predicted = ", linear.predict(X_test))
print("linear mean_squared_error = ", mean_squared_error(linear.predict(X_test), y_test))


# In[11]:


random_forest_regressor = RandomForestRegressor()
random_forest_regressor.fit(X_train, y_train)
# print("random_forest_regressor Predicted = ", random_forest_regressor.predict(X_test))
print("random_forest_regressor mean_squared_error = ", mean_squared_error(random_forest_regressor.predict(X_test), y_test))


# In[12]:


plt.figure()
plt.scatter(X_test[:, 1], linear.predict(X_test), label = "Predicted price", color = "red")
plt.scatter(X_test[:, 1], y_test, label = "Real price", color = "green")
plt.legend()
plt.show()


# In[ ]:


while True:
    print("please input the two parameters" )
    a = float(input("First parameters YrSold: "))
    b = float(input("Second parameters LotArea: "))
    character = [[a, b]]
    print("linear.predict = ", linear.predict(character)[0])
    print("random_forest_regressor.predict = ", random_forest_regressor.predict(character)[0])


# In[ ]:





# In[ ]:




