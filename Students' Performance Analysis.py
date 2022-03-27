#!/usr/bin/env python
# coding: utf-8

# In[1]:


conda install pandas


# In[2]:


conda install nbconvert


# # Import Library

# In[2]:


import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# # Read Data

# In[3]:


print(pd.__version__)
# print(numpy.__version__)


# In[4]:


data = pd.read_csv('StudentsPerformance.csv')
data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# We need to know the value of each column that non numerical, so we can know how much number that we beed to use for convert the variable into numerical

# In[8]:


print("value in gender: ")
data["gender"].value_counts()


# In[9]:


print("value in race/ethnicity: ")
data["race/ethnicity"].value_counts().sort_values(ascending=True)


# In[10]:


print("value in parental level of education: ")
data["parental level of education"].value_counts()


# In[11]:


print("value in lunch: ")
data["lunch"].value_counts()


# In[12]:


print("test preparation course: ")
data["test preparation course"].value_counts()


# # Preprocessing Data

# In[13]:


# show missing data
print("\nMissing Data: ")
print(data.isnull().sum())

#drop NAN data
data.dropna()

#show duplicate data
print("\n=================================")
print("\nData Duplicated: ", data.duplicated().sum())


# In[14]:


#change non numerical data into numerical

encoding = {"gender": {"male":0, "female": 1},
           "race/ethnicity": {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E":4},
           "parental level of education": {"some college": 0, "associate's degree": 1, "high school": 2, "some high school": 3, "bachelor's degree": 4, "master's degree": 5},
           "lunch": {"standard": 0, "free/reduced": 1},
           "test preparation course": {"none": 0, "completed": 1}}
data.replace(encoding, inplace = True)

data.head()


# In[15]:


#Binning Data

score = [0, 40, 70, 100]
group_names = ["low", "medium", "high"]
data["math score group"] = pd.cut(data['math score'], score, labels = group_names)

print(data["math score group"].value_counts())
data.head()


# # Visualise data

# In[16]:


#visualize gender
label_gender = ["male", "female"]
male = data['gender'].value_counts()[0]
female = data['gender'].value_counts()[1]

gender = [male, female]
colors = ['#66b3ff','#ff9999']

plt.title("Gender Percentage")
plt.pie(gender, labels = label_gender, colors = colors, autopct = '%1.2f%%')
plt.show()


# In[17]:


#visualize race/ethnicity

label_race = ["Group A", "Group B", "Group C", "Group D", "Group E"]
race = [data['race/ethnicity'].value_counts()[0], 
        data['race/ethnicity'].value_counts()[1], 
        data['race/ethnicity'].value_counts()[2],
        data['race/ethnicity'].value_counts()[3],
        data['race/ethnicity'].value_counts()[4]]
colors = ['#7C99AC', '#406882', '#6998AB', '#B1D0E0', '#D3DEDC']

plt.title("Race/Ethnicity Percentage")
plt.pie(race, labels = label_race, colors = colors, autopct = '%1.2f%%')
plt.show()


# In[18]:


#visualize parent level education

label_parent_education = ["some college", 
                          "associate's degree", 
                          "high school", 
                          "some high school", 
                          "bachelor's degree", 
                          "master's degree"]
parent_education = [data["parental level of education"].value_counts()[0],
                   data["parental level of education"].value_counts()[1],
                   data["parental level of education"].value_counts()[2],
                   data["parental level of education"].value_counts()[3],
                   data["parental level of education"].value_counts()[4],
                   data["parental level of education"].value_counts()[5]]
colors = ['#FFBD9B', '#865439', '#C68B59', '#D7B19D', '#CC9B6D','#F1CA89']

plt.title("Parent Level Education Percentage")
plt.pie(parent_education, labels = label_parent_education, colors = colors, autopct = '%1.2f%%')
plt.show()


# In[19]:


#visualize test preparation

lunch = [data['lunch'].value_counts()[0],
         data['lunch'].value_counts()[1]]
label_lunch = ['standard', 'free/reduced']
colors = ['#42C2FF', '#85F4FF']

plt.title("Student's Lunch Percantage")
plt.pie(lunch, labels = label_lunch, colors = colors, autopct = '%1.2f%%')
plt.show()


# In[20]:


#visualize test preparation

test_preparation = [data["test preparation course"].value_counts()[0],
                    data["test preparation course"].value_counts()[1]]
label_test_preparation = ["none", "completed"]
colors = ['#7C73E6','#C4C1E0']

plt.title("Students who take test preparation course")
plt.pie(test_preparation, labels = label_test_preparation, colors = colors, autopct = '%1.2f%%')
plt.show()


# In[21]:


#visualize math score
sorted_score = data["math score"].value_counts().sort_index(ascending = False)
colors = ['#142850', '#27496D', '#00909E', '#064663', '#84A9AC']

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("Math Score")
ax1.scatter(x = sorted_score.index, y = sorted_score)
ax1.set_xlabel("Score")
ax1.set_ylabel("Number of Students")

ax2.set_title("Top 5 Math Score")
# ax2.scatter(x = sorted_score.head().index, y = sorted_score.head())
sns.barplot(x = sorted_score.head().index, y= sorted_score.head(), palette = colors)
ax2.set_xlabel("Score")
ax2.set_ylabel("Number of Students")

plt.show()


# In[22]:


#Visualize reading score

reading_score = data['reading score'].value_counts().sort_index(ascending = False)
colors = ["#D0CAB2", "#FFE6BC", "#E4CDA7", "#C3B091", "#8E806A"]

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("reading score")
ax1.scatter(reading_score.index, reading_score)
ax1.set_xlabel("score")
ax1.set_ylabel("number of students")

ax2.set_title("top 5 reading score")
sns.barplot(x = reading_score.head().index, y = reading_score.head(), data = data, palette = colors)
ax2.set_xlabel("score")
ax2.set_ylabel("number of students")
plt.show()


# In[8]:


#visualize writing score

writing_score = data['writing score'].value_counts().sort_index(ascending = False)
colors = ['#051367','#2D31FA','#5D8BF4','#42C2FF','#85F4FF']

fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("writing score")
ax1.scatter(writing_score.index, writing_score)
ax1.set_xlabel("score")
ax1.set_ylabel("number of students")

ax2.set_title("top 5 writing score")
sns.barplot(data = data, x = writing_score.head().index, y = writing_score.head(), palette = colors)
ax2.set_xlabel("score")
ax2.set_ylabel("number of students")

plt.show()


# # CORRELATION

# In[24]:


#show correlation using pearson method
data_correlation = data.corr(method = 'pearson' )
data_correlation


# In[25]:


#visualize correlation in math score
fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("correlation between math score and reading score: {}".format(data_correlation['math score']['reading score']))
ax1.scatter(data['math score'], data['reading score'])
ax1.set_xlabel('math score')
ax1.set_ylabel('reading score')

ax2.set_title("correlation between math score and writing score: {} ".format(data_correlation['math score']['writing score']))
ax2.scatter(data['math score'], data['writing score'])
ax2.set_xlabel('math score')
ax2.set_ylabel('writing score')

plt.show()


# In[26]:


#visualize correlation in reading score

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("correlation between reading score and math score: {}".format(data_correlation['reading score']['math score']))
ax1.scatter(data['reading score'], data['math score'])
ax1.set_xlabel("reading score")
ax1.set_ylabel("math score")

ax2.set_title("correlation between reading score and writing score {}"
              .format(data_correlation['reading score']['writing score']))
ax2.scatter(data['reading score'], data['writing score'])
ax2.set_xlabel('reading score')
ax2.set_ylabel('writing score')

plt.show()


# In[27]:


#visualize correlation in writing score

fig = plt.figure(figsize=(15, 5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("correlation between writing score and math score: {}".format(data_correlation['writing score']['math score']))
ax1.scatter(data['writing score'], data['math score'])
ax1.set_xlabel("writing score")
ax1.set_ylabel("math score")

ax2.set_title("correlation between writing score and reading score: {}"
              .format(data_correlation['writing score']['reading score']))
ax2.scatter(data['writing score'], data['reading score'])
ax2.set_xlabel("writing score")
ax2.set_ylabel("reading score")
plt.show()

