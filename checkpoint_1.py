### STEP_2__. Import required libraries (not allowed: scikit-learn or any other libraries with inbuilt functions that help to implement ML)

#all required libraries imported 
#import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  
import seaborn as sns 
import time

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly

### STEP_1__Select one dataset from the list provided above. The datasets are located in the folder “dataset”, use only the dataset provided in the folder

#reading the File
#pd.set_option('display.max_rows', None)
data_diamond_3=pd.read_csv("C:/Users/viraj/Code_a/ML_Prof_Alina/Assignment_1/datasets (1)/datasets/diamond.csv")

### STEP_3__Read, preprocess and print the main statistic about the dataset (your code from Part I can be reused).

data_diamond_3.head()

#checking the null values
data_diamond_3.isnull().sum() 

data_diamond_3.info()

data_diamond_3 = data_diamond_3.drop(["Unnamed: 0"], axis=1)

data_diamond_3

df=data_diamond_3
#printing the column names
for col in df.columns:
    print(col)

df.head().transpose()
df = pd.DataFrame(df)

df.drop(df[df['x'] == 0].index, inplace = True)
df.drop(df[df['y'] == 0].index, inplace = True)
df.drop(df[df['z'] == 0].index, inplace = True)

df

print(f'cut_unique_Values_{ df["cut"].unique() }')
print(f'clarity_unique_values_{df["clarity"].unique()}')
print(f'color_unique_values_{df["color"].unique()}')

df
df_1=df


numeric_clarity= {"clarity": {"I1":0,"SI2":1,"SI1":2,"VS2":3, "VS1":4, "VVS2":5, "VVS1":6,"IF":7}}
df_1 = df_1.replace(numeric_clarity)

numeric_color = {"color": {"J":0,"I":1,"H":2,"G":3, "F":4, "E":5, "D":6}}
df_1 = df_1.replace(numeric_color)

numeric_cut = {"cut": {"Fair":0, "Good":1, "Very Good":2, "Premium":3, "Ideal":4}}
df_1 = df_1.replace(numeric_cut)

df_1

data_diamond_3.info()

df_1['cut'] = df_1['cut'].astype('int64')
df_1['color'] = df_1['color'].astype('int64')
df_1['clarity'] = df_1['clarity'].astype('int64')
data_diamond_3.shape

df_1.info()
df_2=df_1

df_1.describe()



#a. Find the min and max values for each column.

x_min , x_max = min(df_1['x']),max(df_1['x'])

y_min ,  y_max = min(df_1['y']) ,max(df_1['y'])

z_min ,z_max = min(df_1['z']) , max(df_1['z'])
 
carat_min , carat_max = min(df_1['carat']) , max(df_1['carat'])


depth_min , depth_max = min(df_1['depth']),max(df_1['depth'])

table_min ,  table_max = min(df_1['table']) ,max(df_1['table'])

price_min ,price_max = min(df_1['price']) , max(df_1['price'])

#b. Rescale dataset columns to the range from 0 to 1

df_1['x'] = (df_1['x'] - x_min) / (x_max - x_min)
df_1['y']  = (df_1['y']  - y_min)  / (y_max  - y_min)
df_1['z'] = (df_1['z'] - z_min)/(z_max - z_min)
df_1['carat'] = (df_1['carat'] - carat_min)/(carat_max - carat_min)



df_1['depth'] = (df_1['depth'] - depth_min) / (depth_max - depth_min)
df_1['table']  = (df_1['table']  - table_min)  / (table_max  - table_min)
df_1['price'] = (df_1['price'] - price_min)/(price_max - price_min)

## 

import random
df_1 = df_1.sample(frac = 1)

df_1_except_price_X=df_1[['x','y','z','carat']]
df_1_except_price_X.shape

df_1_price_Y=df_1['price']
print(df_1_price_Y.shape)

df_1_price_Y

### STEP_8__Divide the dataset into training and test, as 80% training, 20% testing dataset
### STEP_6__Choose your target Y
### STEP_7__Create the data matrices for X (input) and Y (target) in a shape X = 𝑁 x 𝑑 and Y = 𝑁 x 1, where 𝑁 is a number of data samples and 𝑑 is a number of features.


import random
random.seed(46)

train_size = int(len(df_1_except_price_X) * 0.8)

X_train = df_1_except_price_X[0:train_size]
Y_train = df_1_price_Y[0:train_size]

X_test = df_1_except_price_X[train_size : ]
Y_test = df_1_price_Y[train_size : ]

X_train = X_train.to_numpy()
Y_train = Y_train.to_numpy()
X_test = X_test.to_numpy()
Y_test = Y_test.to_numpy()


### STEP_9__Print the shape of your X_train, y_train, X_test, y_test.

print(X_train.shape)
print(Y_train.shape)

print(X_test.shape)
print(Y_test.shape)

Use your implementation from Part II and extend it to Ridge Regression.
Implement parameter estimation for ridge regression by minimizing the regularized
squared loss as follows:
    
   #### 𝐽(𝒘) = (0.5) * (∑ <i=1 to N> ((𝑦_𝑖 − (𝑤^𝑇 ) * 𝑥_𝑖 )^2))  +  (0.5) * (𝜆 * (𝒘^𝑇) * 𝒘) 
    
    In matrix-vector notation, the squared loss can be written as:
   #### 𝐽(𝒘) = (0.5) * (∑ <i=1 to N> ((𝒚 − 𝑿 * 𝒘)^𝑇) * (𝒚 − 𝑿 * 𝒘))  +  (0.5) * (𝜆 * (𝒘^𝑇) * 𝒘)   
    
    OLS equation for Ridge regression can be estimated as
   #### 𝒘 = (((𝑿^𝑇) * 𝑿 + 𝝀 * 𝑰)^(−1)) * (𝑿^𝑇) * y




class RidgeRegression_with_GradientDescent():
    
    def __init__(self, l2_regularizer=0.1, learning_rate=1e-6, epochs=10000):
        self.weights = None
        self.l2_regularizer = l2_regularizer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.cost_history = []
        self.gradient_history = []

    def fit(self):
        inverse_term = np.linalg.inv((np.matmul(X_train.T, X_train)) + self.l2_regularizer * np.identity(X_train.shape[1]))
        self.weights = np.matmul(np.matmul(inverse_term, X_train.T), Y_train)     
        for i in range(self.epochs):
            self.cost_history.append(self.cost(Y_train, X_train, self.weights))
            gradient = self.gradient(Y_train, X_train, self.weights)
            self.gradient_history.append(gradient)
            self.weights -= self.learning_rate * gradient
        return self.weights

    def gradient(self, Y_train, X_train, weights):
        gradient = -2 * np.matmul(X_train.T, (Y_train - np.matmul(X_train, self.weights))) + 2 * self.l2_regularizer * self.weights
        return gradient

    def cost(self, Y_train, X_train, weights):
        residual = np.matmul((Y_train - np.matmul(X_train, self.weights)).T, (Y_train - np.matmul(X_train, self.weights))) + self.l2_regularizer * np.matmul(self.weights.T, self.weights)
        return residual/len(Y_train)

    def predict(self, X):
        Y_pred = np.matmul(X, self.weights)
        return Y_pred


# initialize the class object
model = RidgeRegression_with_GradientDescent()

# fit the model and get weights vector
# fit the model and get weights vector


start_time = time.perf_counter()
weights = model.fit()
end_time = time.perf_counter()
training_time = end_time - start_time
print(f"Time taken to train the model: {training_time:.2f} seconds")
weights_list = weights.tolist()
print("Model weights:", weights_list)

# Cost_Function_Training test
cost_function_train = model.cost(Y_train, X_train, weights)
cost_function_train

# Cost_Function_testing test
cost_function_test = model.cost(Y_test, X_test, weights)
cost_function_test

Y_pred_test = model.predict( X_train)
Y_pred_test

# predict the output on test set
Y_test_prediction = model.predict(X_test)
Y_test_prediction

# compare model using inbuilt sklearn function
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
clf = Ridge(alpha=0.1,solver='sag')
x = clf.fit(X_train, Y_train)
# predict the output for the train set
y_pred_inbuiltfuntioon = clf.predict(X_train)
mse1 = mean_squared_error(Y_train,y_pred_inbuiltfuntioon)
print(f'The cost of inbuilt function is {round(mse1,2)}')



### STEP_11__Get the predictions and calculating the sum of squared errors:


# calculate the rmse of train set
rmse = np.sqrt(cost_function_train)
rmse

# calculate the rmse of test set
rmse = np.sqrt(cost_function_test)
rmse

### STEP_12__Plot the predictions vs the actual targets


trace_pred = go.Scatter(x=np.arange(len(Y_train_prediction)), y=Y_train_prediction.reshape(-1),
                        mode='markers', name='Predicted')

trace_actual = go.Scatter(x=np.arange(len(Y_train)), y=Y_train.reshape(-1),
                          mode='markers', name='Actual', marker=dict(color='red'))

data = [trace_pred, trace_actual]

layout = go.Layout(title='Predicted vs TARGET value for training_set',legend=dict(x=0, y=1))


fig = go.Figure(data=data, layout=layout)
fig.show()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=list(range(len(Y_test_prediction))),
    y=Y_test_prediction,
    mode='markers',
    name='Predicted'
))

fig.add_trace(go.Scatter(
    x=list(range(len(Y_test))),
    y=Y_test,
    mode='markers',
    name='Actual'
))

fig.update_layout(
    title='Predicted vs TARGET value for testing_set',
    legend=dict(
        x=0,
        y=1,
        bgcolor='rgba(255, 255, 255, 0.5)',
        bordercolor='rgba(0, 0, 0, 0.5)',
        borderwidth=1
    )
)
fig.show()

