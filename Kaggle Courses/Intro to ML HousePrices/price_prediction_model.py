# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 11:44:01 2024

@author: mark
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Path of the file to read, it is in the same folder as this file so you only need the file name
file = 'housing.csv'

# Fill in the line below to read the file into a variable data
data = pd.read_csv(file)

#print(data.columns)

# Print summary statistics in next line
#print(data.describe())
#print(data.bedrooms.describe())
#print(data.bathrooms.describe())


# What is the average square footage (rounded to nearest integer)?
avg_sq_ft = data.area.mean()
#5151 square feet
#print(int(avg_sq_ft.round()))

# What is the average number of bedrooms (rounded to nearest integer)?
avg_bedrooms = data.bedrooms.mean()
#3
#print(int(avg_bedrooms.round()))

"""Desicion Tree Machine Learning Model"""
# Set prediction target
y=data.price

# Set features
features = ['area', 'bedrooms', 'bathrooms', 'stories']
X = data[features]

# Look at numerical features
#print(X.describe())
#print(X.head)

# Define model. Specify a number for random_state to ensure same results each run
housing_model = DecisionTreeRegressor(random_state=1)

# Fit model
housing_model.fit(X, y)

# Test the prediction process, we will do a more appropraite test later
#print("Making predictions for the following 5 houses:")
#print(X.head())
#print("The predictions are")
#print(housing_model.predict(X.head()))
# Actual prices
#print(y[:5])

# Mean absolute error = 145,165
predicted_home_prices = housing_model.predict(X)
#print(mean_absolute_error(y, predicted_home_prices))


"""Validate Model"""
# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)
# Define model
housing_model = DecisionTreeRegressor()
# Fit model
housing_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = housing_model.predict(val_X)
# Mean absolute error is approx. 1,300,000
#print(mean_absolute_error(val_y, val_predictions))


"""Underfitting/Overfitting"""
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# compare MAE with differing values of max_leaf_nodes
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
maes = []
for max in candidate_max_leaf_nodes:
    maes.append(get_mae(max, train_X, val_X, train_y, val_y))
#print(maes)    

# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
best_tree_size = candidate_max_leaf_nodes[maes.index(min(maes))]
#print(best_tree_size)

# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state=2)

# fit the final model and uncomment the next two lines
final_model.fit(X, y)


"""Random Forest"""
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state = 1)

# fit your model
rf_model.fit(train_X, train_y)

# Calculate the mean absolute error of your Random Forest model on the validation data
val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(val_y, val_predictions)

#print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

"""Create a model for Competition Submission"""
# Select columns corresponding to features, and preview the data
X = data[features]
X.head()

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

#print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data
rf_model_on_full_data = RandomForestRegressor(random_state=2)

# fit rf_model_on_full_data on all data from the training data
rf_model_on_full_data.fit(X, y)
rf_model_on_full_data_val_predictions = rf_model_on_full_data.predict(val_X)
rf_model_on_full_data_val_mae = mean_absolute_error(rf_model_on_full_data_val_predictions, val_y)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = data[features]

# make predictions which we will submit. 
test_preds = rf_model_on_full_data.predict(test_X)

#print(test_preds)