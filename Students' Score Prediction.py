#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
import sklearn.metrics  as met
from sklearn.metrics import r2_score
import numpy as np
from sklearn.preprocessing import MinMaxScaler

get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('StudentsPerformance.csv')

data.head()


# In[3]:


data.tail()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


print('missing data: ')
data.isnull().sum()


# In[7]:


print('Duplicated data: ', data.duplicated().sum())


# In[8]:


#change non numerical data into numerical

encoding = {"gender": {"male":0, "female": 1},
           "race/ethnicity": {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E":4},
           "parental level of education": {"some college": 0, "associate's degree": 1, "high school": 2, "some high school": 3, "bachelor's degree": 4, "master's degree": 5},
           "lunch": {"standard": 0, "free/reduced": 1},
           "test preparation course": {"none": 0, "completed": 1}}
data.replace(encoding, inplace = True)

data.head()


# In[9]:


data.corr(method = 'pearson')


# # Prediction Students' Performance in Math Score

# In[10]:


#seperate data
x_for_math_score = data.drop(columns = ['math score'])
y_math_score = data['math score']

#Divide data into train and test for writing score
x_for_math_score_train, x_for_math_score_test, y_math_score_train, y_math_score_test = tts(x_for_math_score, y_math_score, test_size = 0.3, random_state=1, shuffle = True)


# In[11]:


#Visualize the target's data

plt.title("Math Score")
plt.scatter(y_math_score.value_counts(), y_math_score.value_counts().index)
plt.xlabel("Number of Students who have the Score")
plt.ylabel("Score")
plt.show()


# In[12]:


print("X train shape: {} and X test shape: {}".format(x_for_math_score_train.shape, x_for_math_score_test.shape))
print("\n==========================================")
print("\nY train shape: {} and Y test shape: {}".format(y_math_score_train.shape, y_math_score_test.shape))


# In[13]:


model_math = LinearRegression()
model_math.fit(x_for_math_score_train, y_math_score_train)


# In[14]:


coefficient = model_math.coef_
coefficient


# In[15]:


intercept = model_math.intercept_
intercept


# ## Predict and Evaluate Prediction for Math Score Exam

# In[16]:


y_predict = model_math.predict(x_for_math_score_test)


# In[17]:


print("Predict Accuracy: ", r2_score(y_math_score_test, y_predict))


# #### Using the multiple linear regression equation: Y = (coefficient1 * x1 + coefficient2 * x2 +... coefficient * x) + intercept. We will get the prediction's value as below:

# In[18]:


x_test = np.array(x_for_math_score_test.iloc[0])
print(x_test)
print("\nPrediction Value using Predict module: ", y_predict[0])
print("\nPrediction Value with Equation: ", (x_test @ coefficient) + intercept)


# ## Visualize Math Score Prediction

# In[19]:


plt.scatter(y_math_score_test, y_predict)

plt.title("Math Score Prediction")
plt.xlabel("Actual Score")
plt.ylabel("Predict Score")
plt.show()


# # Prediction Students' Performance in Reading Score

# In[20]:


#seperate data
x_for_reading_score = data.drop(columns = ['reading score'])
y_reading_score = data['reading score']

#Divide data into train and test for math score
x_for_reading_score_train, x_for_reading_score_test, y_reading_score_train, y_reading_score_test = tts(x_for_reading_score,y_reading_score, test_size = 0.3, random_state=1, shuffle = True)


# In[21]:


#Visualize the target's data

plt.title("Reading Score")
plt.scatter(y_reading_score.value_counts(), y_reading_score.value_counts().index)
plt.xlabel("Number of Students who have the Score")
plt.ylabel("Score")
plt.show()


# In[22]:


print("X train shape: {} and X test shape: {}".format(x_for_reading_score_train.shape, x_for_reading_score_test.shape))
print("\n==========================================")
print("\nY train shape: {} and Y test shape: {}".format(y_reading_score_train.shape, y_reading_score_test.shape))


# In[23]:


model_reading = LinearRegression()
model_reading.fit(x_for_reading_score_train, y_reading_score_train)


# In[24]:


coefficient = model_reading.coef_
coefficient


# In[25]:


intercept = model_reading.intercept_
intercept


# ## Predict and Evaluate Prediction for Math Score Exam

# In[26]:


y_predict = model_reading.predict(x_for_reading_score_test)


# In[27]:


print("Predict Accuracy: ", r2_score(y_reading_score_test, y_predict))


# #### Using the multiple linear regression equation: Y = (coefficient1 * x1 + coefficient2 * x2 +... coefficient * x) + intercept. We will get the prediction's value as below:

# In[28]:


x_test = np.array(x_for_reading_score_test.iloc[0])
print(x_test)
print("\nPrediction Value using Predict module: ", y_predict[0])
print("\nPrediction Value with Equation: ", (x_test @ coefficient) + intercept)


# ## Visualize Reading Score Prediction

# In[29]:


plt.scatter(y_reading_score_test, y_predict)
plt.title("Reading Score Prediction")
plt.xlabel("Actual Score")
plt.ylabel("Predict Score")
plt.show()


# # Prediction Students' Performance in Writing Score

# In[30]:


#seperate data
x_for_writing_score = data.drop(columns = ['writing score'])
y_writing_score = data['writing score']

#Divide data into train and test for writing score
x_for_writing_score_train, x_for_writing_score_test, y_writing_score_train, y_writing_score_test = tts(x_for_writing_score, y_writing_score, test_size = 0.3, random_state=1, shuffle = True)


# In[31]:


#Visualize the target's data

plt.title("Writing Score")
plt.scatter(y_writing_score.value_counts(), y_writing_score.value_counts().index)
plt.xlabel("Number of Students who have the Score")
plt.ylabel("Score")
plt.show()


# In[32]:


print("X train shape: {} and X test shape: {}".format(x_for_writing_score_train.shape, x_for_writing_score_test.shape))
print("\n==========================================")
print("\nY train shape: {} and Y test shape: {}".format(y_writing_score_train.shape, y_writing_score_test.shape))


# In[39]:


model_writing = LinearRegression()
model_writing.fit(x_for_writing_score_train, y_writing_score_train)


# In[40]:


coefficient = model_writing.coef_
coefficient


# In[41]:


intercept = model_writing.intercept_
intercept


# In[42]:


y_predict = model_writing.predict(x_for_writing_score_test)


# In[43]:


print("Predict Accuracy: ", r2_score(y_writing_score_test, y_predict))


# #### Using the multiple linear regression equation: Y = (coefficient1 * x1 + coefficient2 * x2 +... coefficient * x) + intercept. We will get the prediction's value as below:

# In[44]:


x_test = np.array(x_for_writing_score_test.iloc[0])
print(x_test)
print("\nPrediction Value using Predict module: ", y_predict[0])
print("\nPrediction Value with Equation: ", (x_test @ coefficient) + intercept)


# ## Visualize Math Score Prediction

# In[45]:


plt.scatter(y_reading_score_test, y_predict)
plt.title("Reading Score Prediction")
plt.xlabel("Actual Score")
plt.ylabel("Predict Score")
plt.show()

