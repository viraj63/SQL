#!/usr/bin/env python
# coding: utf-8

# ## Step 1 - Import Libraries.
# 
# Import required libraries (not allowed: scikit-learn or any other libraries with inbuilt functions that help to implement ML methods).

# In[1]:


#all required libraries imported 
import numpy as np
import pandas as pd 


# ## Step 2 - Reading the data and printing the statistics.

# In[2]:


#reading the File
#pd.set_option('display.max_rows', None)
data=pd.read_csv("C:/Users/viraj/Code_a/ML_Prof_Alina/Assignment_1/datasets (1)/datasets/penguins.csv")


# Read, preprocess, and print the main statistics about the dataset (you can reuse
# your code from Assignment 0 with a proper citation)
# 

# In[3]:


#showcasing the first 5  rows of the dataset
data.head()


# In[4]:


#checking the null values
data.isnull().sum() 


# In[5]:


#deleting the rows having NAN values
data=data.dropna()


# In[6]:


#showcasing the Datatype of elements of the columns
data.info()


# In[7]:


#The main statistics of the dataSet
data.describe()


# In[8]:


d1=data.species.describe()   
d2=data.island.describe() 
d3=data.sex.describe() 
print(d1,d2,d3)


# In[9]:


#printing the column names
for col in data.columns:
    print(col)  


# In[ ]:





# ## STEP 3 - Convert features with string datatype to categorical (species, island, sex).
# 
# Example: suppose you have a dataset that contains information about movies,
# with the following features: title (string), director (string), genre (string). You need
# to convert these features of string datatype to categorical features. This can be
# done by assigning a unique numerical value to each unique string value in each
# categorical feature.

# In[10]:


# Here Species,islands and sex has datatype as string 

data[['species','island','sex']] = data[['species','island','sex']].astype('category')


# In[11]:


data.info()


# In[12]:


# Converting Categorical value into numerical value
data['sex'] = pd.factorize(data['sex'])[0]
data['island'] = pd.factorize(data['island'])[0]
data['species'] = pd.factorize(data['species'])[0]


# In[13]:


# Here we could see the datatype has been changed of the following 
data.info() 


# In[14]:


print(data.dtypes)


# In[15]:


data.shape


# ## 4. Normalize non-categorical features (bill_length_mm, bill_depth_mm,flipper_length_mm, body_mass_g).
# 
# a. Find the min and max values for each column.
# 
# b. Rescale dataset columns to the range from 0 to 1
# 
# 
# Why do we do this? Normalization is to transform features to be on a similar
# scale. This improves the performance and training stability of the model.
# 
# ###### Note: normalize() is not allowed as it is a part of scikit-learn library.

# In[16]:


data.head()


# In[17]:


#a. Find the min and max values for each column.

bill_length_mm_min , bill_length_mm_max = min(data['bill_length_mm']),max(data['bill_length_mm'])

bill_depth_mm_min ,  bill_depth_mm_max = min(data['bill_depth_mm']) ,max(data['bill_depth_mm'])

flipper_length__mm_min ,flipper_length_mm_max = min(data['flipper_length_mm']) , max(data['flipper_length_mm'])
 
body_mass_g_min , body_mass_g_max = min(data['body_mass_g']) , max(data['body_mass_g'])
 


# In[18]:


data1=data


# In[19]:


#b. Rescale dataset columns to the range from 0 to 1

data1['bill_length_mm'] = (data1['bill_length_mm'] - bill_length_mm_min) / (bill_length_mm_max - bill_length_mm_min)
data1['bill_depth_mm']  = (data1['bill_depth_mm']  - bill_depth_mm_min)  / (bill_depth_mm_max  - bill_depth_mm_min)
data1['flipper_length_mm'] = (data1['flipper_length_mm'] - flipper_length__mm_min)/(flipper_length_mm_max - flipper_length__mm_min)
data1['body_mass_g'] = (data1['body_mass_g'] - body_mass_g_min)/(body_mass_g_max - body_mass_g_min)


# In[20]:


data1


# ## 5. Choose your target Y. For this dataset, there are several options:
# a. We can use a binary classifier to predict which gender a penguin belongs to (female or male). In this case, column sex can be used as Y (target)
# 
# 
# b. We can use a binary classifier to predict if a penguin’s location is Torgersen island or not. In this case, column island can be used as Y (target) 
# 
# 

# In[21]:


# Step 5 - Choosing target
# Here we have chosen sex as the target, rest all are the inputs some of them are dropped.


# ## STEP 6 - Create the data matrices for X (input) and Y (target) in a shape,X = 𝑁 x 𝑑 and Y = 𝑁 x 1, were 𝑁 is a number of data samples and 𝑑 has a number of features. 
# 

# In[22]:


data1


# In[23]:


data


# In[24]:


data1['sex'] = pd.factorize(data1['sex'])[0]
data1['island'] = pd.factorize(data1['island'])[0]
data1['species'] = pd.factorize(data1['species'])[0]


# In[25]:


data1.info()


# In[26]:


import random
data1 = data1.sample(frac = 1)


# In[27]:


data2_except_sex_X=data1[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']]
data2_except_sex_X.shape


# In[28]:


data3_sex_column_Y=data1['sex']
print(data3_sex_column_Y.shape)
data3_sex_column_Y=data3_sex_column_Y.astype(int)
#print(data3_sex_column_Y.dtype)


# ## Step 7 - Divide the dataset into training and test, as 80% training, 20% testing dataset.
# 

# In[29]:


import random
random.seed(46)

train_size = int(len(data2_except_sex_X) * 0.8)

X_train = data2_except_sex_X[0:train_size]
Y_train = data3_sex_column_Y[0:train_size]

X_test = data2_except_sex_X[train_size : ]
Y_test = data3_sex_column_Y[train_size : ]


# ## Step 8 -Print the shape of your X_train, y_train, X_test, y_test
# 

# In[30]:


print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)


# ## Step 9 - Code for the Logistic Regression using the recommended structure of the code for defining logistic regression:
# 

# In[31]:


import math
class LogitRegression():
    
    def __init__(self, learning_rate = 0.001, iterations = 10000):
        # Takes as an input hyperparameters: learning rate and the number of iterations
        # Has weights and bias also.
        # We have self.losses to append the losses.
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.losses = []
        self.weights = None
        self.bias = None
        
    
    def sigmoid(self, x):
        # Defining the sigmoid function.
        sigma = 1/(1 + np.exp(-x))
        return sigma
    
    def cost(self, y, y_pred):
        # Defining the loss function.
        N = len(y)
        # Formula for the loss.
        cost = -(1/N) * (y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
        return cost
        
    def gradient(self, n, X, y, y_pred):
        # Defining gradient function.
        delta = y_pred - y
        # Formula for the weights.
        dw = (1 / n) * np.dot(np.transpose(X), (delta))
        # Formula for the bias.
        db = (1 / n) * np.sum(delta)
        return dw, db
    
    def fit(self, X, y):
        n, no_feature  = X.shape
        # Assigning random weights and bias zero.
        self.weights = np.random.uniform(0, 1, 4)
        self.bias = 0
        for i in range(self.iterations):
            # Using sigmoid defined to get prediction.
            y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
            # Getting weights and bias.
            dw, db = self.gradient(n, X, y, y_pred)
            # Calculating loss
            loss = self.cost(y, y_pred)
            print(f"Iteration {i}: loss = {np.mean(loss)}")
            # Appending loss to the list.
            self.losses.append(loss)
            # Update weights and bias
            self.weights -= self.learning_rate*dw
            self.bias -= self.learning_rate*db
    
    def predict(self, X):
        y_pred = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_sex = []
        for i in y_pred:
            # Appending 1 if i is greater than or equal to 0.5.
            if i >= 0.5:
                y_sex.append(1)
            # Appending 0 if i is less than 0.5.
            else:
                y_sex.append(0)
        weight = self.weights
        return y_sex, weight


# ## Step 10 -  Train the model:
# a. Define a model by calling LogitRegression class and passing
# hyperparameters, e.g.
# model = LogitRegression(learning_rate, iterations)
# b. Train the model, by calling fit function and passing your training dataset,
# e.g
# model.fit(X_train, y_train)
# c. Suggested hyperparameters:
# Note: You can try different learning rates and number of iterations to
# improve your accuracy (accuracy of greater than 64% is expected)
# learning_rate=1e-3
# iterations=100000
# weights = np.random.uniform(0, 1)
# 

# In[32]:


import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

model = LogitRegression(0.001, 100000)
model.fit(X_train, Y_train)
plt.plot(model.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.show()


# ## Step 11 - Make a prediction on test dataset by counting how many correct/incorrect predictions your model makes and print your accuracy
# 

# In[33]:


y_pred, weight = model.predict(X_test)
print(y_pred)


# In[34]:


accuracy_lst = []
weight_lst = []
accuracy = np.mean(y_pred == Y_test)
print(accuracy)
accuracy_lst.append(accuracy)
weight_lst.append(weight)


# ## Fitting for different values of learning rate and iterations.

# ### Case 2

# In[ ]:


import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

model1 = LogitRegression(0.0005, 100000)
model1.fit(X_train, Y_train)
plt.plot(model1.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.show()


# In[ ]:


y_pred, weight = model1.predict(X_test)


# In[ ]:


accuracy = np.mean(y_pred == Y_test)
print(accuracy)
print(weight)
accuracy_lst.append(accuracy)
weight_lst.append(weight)
print(weight_lst)


# ### Case 3

# In[ ]:


import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

model2 = LogitRegression(0.1, 10000)
model2.fit(X_train, Y_train)
plt.plot(model2.losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss during training')
plt.show()


# In[ ]:


y_pred, weight = model2.predict(X_test)


# In[ ]:


accuracy = np.mean(y_pred == Y_test)
print(accuracy)
print(weight)
accuracy_lst.append(accuracy)
weight_lst.append(weight)


# In[ ]:


max_accuracy = max(accuracy_lst)
i = accuracy_lst.index(max_accuracy)
weight_max = weight_lst[i]


# In[ ]:


import pickle
pickle.dump( weight_max, open( "weight_pickle.p", "wb" ) )


# In[ ]:


weight_pickle = pickle.load( open( "weight_pickle.p", "rb" ) )


# In[ ]:


print(f'Weights for best accuracy {weight_pickle}')


# 

# Part II: Linear Regression 

# In this part, we implement linear regression model and apply this model to solve a
# problem based on the real-world dataset.
# Datasets that can be used for this part (provided in the zip folder):
# • Flight price prediction dataset
# • Breeding Bird Atlas
# • Diamond dataset (Note: x, y and z columns refer to length, width, and depth
# respectively)
# • Emissions by Country dataset
# • Epicurious – Recipes with Rating and nutrition
# Implement linear regression using the ordinary least squares (OLS) method to perform
# direct minimization of the squared loss function.
# 𝐽(𝒘) =
# 1
# 2
# ∑(𝑦𝑖 − 𝑤
# 𝑇𝑥𝑖
# )
# 2
# 𝑁
# 𝑖=1
# In matrix-vector notation, the loss function can be written as:
# 𝐽(𝒘) =
# 1
# 2
# ∑(𝒚 − 𝑿𝒘)
# 𝑇
# (𝒚 − 𝑿𝒘)
# 𝑁
# 𝑖=1
# where 𝑿 is the input data matrix, 𝒚 is the target vector, and 𝒘 is the weight vector for
# regression.

# 
# ##### Step_1_Select one dataset from the list provided above. The datasets are located in the folder “dataset”, use only the dataset provided in the folder
# 

# In[225]:


data=pd.read_csv("C:/Users/viraj/Code_a/ML_Prof_Alina/Assignment_1/datasets (1)/datasets/diamond.csv")


# In[ ]:




