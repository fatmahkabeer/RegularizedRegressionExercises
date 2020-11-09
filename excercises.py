

#A problem with linear regression is that estimated coefficients of the model can become large,
#making the model sensitive to inputs and possibly unstable.
#This is particularly true for problems with few observations (samples) or less samples (n)
# than input predictors (p) or variables (so-called p >> n problems).
#One approach to address the stability of regression models
#is to change the loss function to include additional costs for a model that has large coefficients.
#Linear regression models that use these modified loss functions during training are
# referred to collectively as penalized linear regression.
# 


#It is known that the ridge penalty shrinks the coefficients
#of correlated predictors towards each other
#while the lasso tends to pick one of them and discard the others. 


#%%  Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.model_selection import train_test_split, cross_val_score 

from statistics import mean 



#%% Loading the Data
data = pd.read_csv('Hitters.csv')

# Drop any rows the contain missing values, along with the player names
data = data.dropna()

# Dropping the numerically non-sensical variables 
dropColumns = ['Unnamed: 0', 'Division', 'League', 'NewLeague' ] 
data = data.drop(dropColumns, axis = 1)

#%% Separating the dependent and independent variables 
y = data['Salary'] 
X = data.drop('Salary', axis = 1) 

# Dividing the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) 


#%% Ridge Regression:
# List to maintain the different cross-validation scores 
cross_val_scores_ridge = [] 

# List to maintain the different values of alpha 
#alpha: Regularization strength;
#must be a positive float.
#Regularization improves the conditioning of the problem and reduces the variance of the estimates.
alpha = [] 

# Loop to compute the different values of cross-validation scores 
for i in range(1, 9): 
	ridgeModel = Ridge(alpha = i * 0.25) 
	ridgeModel.fit(X_train, y_train) 
	scores = cross_val_score(ridgeModel, X, y, cv = 10) 
	avg_cross_val_score = mean(scores)*100
	cross_val_scores_ridge.append(avg_cross_val_score) 
	alpha.append(i * 0.25) 

# Loop to print the different values of cross-validation scores 
for i in range(0, len(alpha)): 
	print(str(alpha[i])+' : '+str(cross_val_scores_ridge[i])) 


# %%# Building and fitting the Ridge Regression model 
ridgeModelChosen = Ridge(alpha = 1) 
ridgeModelChosen.fit(X_train, y_train) 
  
# Evaluating the Ridge Regression model 
print(ridgeModelChosen.score(X_test, y_test)) 

# %%  Lasso Regression:
# List to maintain the cross-validation scores 
cross_val_scores_lasso = [] 

# List to maintain the different values of Lambda 
Lambda = [] 

# Loop to compute the cross-validation scores 
for i in range(1, 9): 
	lassoModel = Lasso(alpha = i * 0.25, tol = 0.0925) 
	lassoModel.fit(X_train, y_train) 
	scores = cross_val_score(lassoModel, X, y, cv = 10) 
	avg_cross_val_score = mean(scores)*100
	cross_val_scores_lasso.append(avg_cross_val_score) 
	Lambda.append(i * 0.25) 

# Loop to print the different values of cross-validation scores 
for i in range(0, len(alpha)): 
	print(str(alpha[i])+' : '+str(cross_val_scores_lasso[i])) 


# %%
# Building and fitting the Lasso Regression Model 
lassoModelChosen = Lasso(alpha = 1, tol = 0.0925) 
lassoModelChosen.fit(X_train, y_train) 

# Evaluating the Lasso Regression model 
print(lassoModelChosen.score(X_test, y_test)) 


# %%
