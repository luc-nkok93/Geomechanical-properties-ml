# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 00:29:45 2024

@author: yvann
"""

new_df2 = new_df1.copy()
new_df2 = new_df2.drop(['DTST', 'NPOR'], axis = 1)

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(new_df2.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

new_df2 = new_df2.drop(['AT10', 'DTCO'], axis = 1)

cols_to_plot = ['DEPT', 'AT90', 'GR', 'HCAL', 'NPHI', 'RHOZ', 'DTSM', 'PR']
sns.pairplot(new_df2[cols_to_plot], diag_kind='kde') ## pairplot for this data computationally expensive?

x = new_df2.iloc[:,0:7].values #independent variables
y = new_df2.iloc[:,7:].values  #dependent variable 

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


from sklearn import metrics
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


# Create an empty list to store the trained DTR models
Dtr_models = []

# Iterate over each column in y_train
for i in range(y_train.shape[1]):
    # Create SVR model
    Dtr = DecisionTreeRegressor(random_state=0)
    # Fit SVR model to the i-th column of y_train
    Dtr.fit(x_train, y_train[:, i])
    # Append trained SVR model to the list
    Dtr_models.append(Dtr)
    
    
# Lists to store performance metrics for each target variable
Dtr_r4_values = []
Dtr_mse4_values = []
Dtr_rmse4_values = []
Dtr_mae4_values = []
y_pred_allDtr = []

# Iterate over each target variable
for i in range(y_test.shape[1]):
    # Make predictions for the current target variable
    y_pred_Dtr = Dtr_models[i].predict(x_test)
    y_pred_allDtr.append(y_pred_Dtr)

    # Compute R2 for the current target variable
    Dtr_r4 = r2_score(y_test[:, i], y_pred_Dtr)
    Dtr_r4_values.append(Dtr_r4)

    # Compute MSE for the current target variable
    Dtr_mse4 = mean_squared_error(y_test[:, i], y_pred_Dtr)
    Dtr_mse4_values.append(Dtr_mse4)

    # Compute RMSE for the current target variable
    Dtr_rmse4 = Dtr_mse4**0.5
    Dtr_rmse4_values.append(Dtr_rmse4)

    # Compute MAE for the current target variable
    Dtr_mae4 = mean_absolute_error(y_test[:, i], y_pred_Dtr)
    Dtr_mae4_values.append(Dtr_mae4)
 
y_pred_allDtr = np.column_stack(y_pred_allDtr)  

for i in range(len(Dtr_r4_values)):
    print(f"Metrics for target variable {i+1}:")
    print(f"R2 Score: {Dtr_r4_values[i]}")
    print(f"Mean Squared Error: {Dtr_mse4_values[i]}")
    print(f"Root Mean Squared Error: {Dtr_rmse4_values[i]}")
    print(f"Mean Absolute Error: {Dtr_mae4_values[i]}")
    print()  

titles = ['PR']
labels = ['Measured PR', 'Predicted PR']

for i in range(min(y_test.shape[1], len(Dtr_models))):
    plt.figure(figsize=(8, 8))
    
    actual_data = y_test[:, i]
    predicted_data = Dtr_models[i].predict(x_test)

    # Check if shapes are compatible for plotting
    if actual_data.shape != predicted_data.shape:
        raise ValueError(f"Shapes of actual_data and predicted_data do not match for target variable {i}")

    # Get the range of the actual data for both axes
    data_range = max(actual_data.max(), predicted_data.max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[0]} - {labels[1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]}')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    
    plt.legend()
    plt.show()
    
    
 ##SCATTER PLOT OF ACTUAL DATA AGAINST PREDICTED DATA FOR THE ENTIRE DATA THAT IS THE PREDICTION WAS DONE ON THE ENTIRE Y

# Lists to store predictions for each target variable
test_Dtr_all = []

# Iterate over each target variable
for i in range(y.shape[1]):
    # Make predictions for the current target variable
    test_Dtr = Dtr_models[i].predict(x)
    test_Dtr_all.append(test_Dtr)

# Now, test_y3_all is a 2D NumPy array with three columns, each representing a target variable
test_Dtr_all = np.column_stack(test_Dtr_all)

for i in range(min(y.shape[1], test_Dtr_all.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y[:, i]
    predicted_data = test_Dtr_all[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y[:, i].max(), test_Dtr_all[:, i].max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]}')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()


#Plot the whole measured data against predicted data
plt.figure(figsize = (15, 5))
plt.plot(new_df2['DEPT'], y[:,0], label='actual PR')
plt.plot(new_df2['DEPT'], test_Dtr_all[:,0], label='Predicted PR')



# Adjust y-axis limits in terms of logarithmic values
plt.ylim(0.0, 0.6)  # Example limits, adjust as needed

# plt.ylim(0, 3)
plt.title("Measured PR VS Predicted PR Using Decision Tree")
plt.xlabel('Depth')
plt.ylabel('DTCO')
plt.legend()
plt.grid()

## EXTRA TREES ALGORITHM  ##

# Create an empty list to store the trained Extra Trees models
extra_trees_models = []

# Iterate over each column in y_train
for i in range(y_train.shape[1]):
    # Create Extra Trees model
    extra_trees = ExtraTreesRegressor(n_estimators=50, random_state=42)
    # Fit Extra Trees model to the i-th column of y_train
    extra_trees.fit(x_train, y_train[:, i])
    # Append trained Extra Trees model to the list
    extra_trees_models.append(extra_trees)
    
    
# Lists to store performance metrics for each target variable
r_ExtraT_values = []
mse_ExtraT_values = []
rmse_ExtraT_values = []
mae_ExtraT_values = []

# Iterate over each target variable
for i in range(y_test.shape[1]):
    # Make predictions for the current target variable
    y_pred_ExtraT = np.column_stack([regressor.predict(x_test) for regressor in extra_trees_models])

    # Compute R2 for the current target variable
    r_ExtraT = r2_score(y_test[:, i], y_pred_ExtraT[:, i])
    r_ExtraT_values.append(r_ExtraT)

    # Compute MSE for the current target variable
    mse_ExtraT = mean_squared_error(y_test[:, i], y_pred_ExtraT[:, i])
    mse_ExtraT_values.append(mse_ExtraT)

    # Compute RMSE for the current target variable
    rmse_ExtraT = mse_ExtraT**0.5
    rmse_ExtraT_values.append(rmse_ExtraT)

    # Compute MAE for the current target variable
    mae_ExtraT = mean_absolute_error(y_test[:, i], y_pred_ExtraT[:, i])
    mae_ExtraT_values.append(mae_ExtraT)
    

# Print the metrics
for i in range(len(r_ExtraT_values)):
    print(f"Metrics for target variable {i+1}:")
    print(f"R2 Score: {r_ExtraT_values[i]}")
    print(f"Mean Squared Error: {mse_ExtraT_values[i]}")
    print(f"Root Mean Squared Error: {rmse_ExtraT_values[i]}")
    print(f"Mean Absolute Error: {mae_ExtraT_values[i]}")
    print()


titles = ['PR']
labels = ['Measured PR', 'Predicted PR']

for i in range(min(y_test.shape[1], len(extra_trees_models))):
    plt.figure(figsize=(8, 8))
    
    actual_data = y_test[:, i]
    predicted_data = extra_trees_models[i].predict(x_test)

    # Check if shapes are compatible for plotting
    if actual_data.shape != predicted_data.shape:
        raise ValueError(f"Shapes of actual_data and predicted_data do not match for target variable {i}")

    # Get the range of the actual data for both axes
    data_range = max(actual_data.max(), predicted_data.max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]} for ExtraTress')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()

##SCATTER PLOT OF ACTUAL DATA AGAINST PREDICTED DATA FOR THE ENTIRE DATA THAT IS THE PREDICTION WAS DONE ON THE ENTIRE Y 
 
# Lists to store predictions for each target variable
test_ExtraT_all = []

# Iterate over each target variable
for i in range(y.shape[1]):
    # Make predictions for the current target variable
    test_ExtraT = extra_trees_models[i].predict(x)
    test_ExtraT_all.append(test_ExtraT)

# Now, test_y3_all is a 2D NumPy array with three columns, each representing a target variable
test_ExtraT_all = np.column_stack(test_ExtraT_all)

for i in range(min(y.shape[1], test_ExtraT_all.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y[:, i]
    predicted_data = test_ExtraT_all[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y[:, i].max(), test_ExtraT_all[:, i].max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]} for ExtraTress')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()


#Plot the whole measured data against predicted data
plt.figure(figsize = (10, 6))
plt.plot(df1['DEPT'], y[:,0], label='actual PR')
plt.plot(df1['DEPT'], test_ExtraT_all[:,0], label='Predicted PR')


# Adjust y-axis limits in terms of logarithmic values
plt.ylim(0.0, 0.6)  # Example limits, adjust as needed

# plt.ylim(0, 3)
plt.title("Actual PR VS Predicted PR Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('PR')
plt.legend()
plt.grid()


## RANDOM FOREST ##

# Create an empty list to store the trained DTR models
rf_models = []

# Iterate over each column in y_train
for i in range(y_train.shape[1]):
    # Create SVR model
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    # Fit SVR model to the i-th column of y_train
    rf.fit(x_train, y_train[:, i])
    # Append trained SVR model to the list
    rf_models.append(rf)


# Define a dictionary of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 150, 175, 200, 225, 250],  # Number of trees in the forest
    'max_features': ['auto', 'sqrt', 'log2'],  # Number of features to consider at every split
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum number of levels in tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'bootstrap': [True, False]  # Method of selecting samples for training each tree
    # Add more hyperparameters to search over if needed
}

# Create an empty list to store the trained Random Forest models
rf_models = []

# Iterate over each column in y_train
for i in range(y_train.shape[1]):
    # Create Random Forest model
    rf_regressor = RandomForestRegressor()
    
    # Perform randomized search for the current target variable
    random_search = RandomizedSearchCV(estimator=rf_regressor, param_distributions=param_grid,
                                       n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=0)
    random_search.fit(x_train, y_train[:, i])
    
    # Get the best estimator from the randomized search
    best_estimator = random_search.best_estimator_
    
    # Append the best estimator to the list of trained Random Forest models
    rf_models.append(best_estimator)
    
    # Print the best parameters and best score for the current target variable
    print(f"Best parameters for target variable {i+1}: {random_search.best_params_}")
    print(f"Best score for target variable {i+1}: {random_search.best_score_}")

r_rf_values = []
mse_rf_values = []
rmse_rf_values = []
mae_rf_values = []

# Iterate over each target variable
for i in range(y_test.shape[1]):
    # Make predictions for the current target variable
    y_pred_rf = np.column_stack([regressor.predict(x_test) for regressor in rf_models])

    # Compute R2 for the current target variable
    r_rf = r2_score(y_test[:, i], y_pred_rf[:, i])
    r_rf_values.append(r_rf)

    # Compute MSE for the current target variable
    mse_rf = mean_squared_error(y_test[:, i], y_pred_rf[:, i])
    mse_rf_values.append(mse_rf)

    # Compute RMSE for the current target variable
    rmse_rf = mse_rf**0.5
    rmse_rf_values.append(rmse_rf)

    # Compute MAE for the current target variable
    mae_rf = mean_absolute_error(y_test[:, i], y_pred_rf[:, i])
    mae_rf_values.append(mae_rf)  

for i in range(len(r_rf_values)):
    print(f"Metrics for target variable {i+1}:")
    print(f"R2 Score: {r_rf_values[i]}")
    print(f"Mean Squared Error: {mse_rf_values[i]}")
    print(f"Root Mean Squared Error: {rmse_rf_values[i]}")
    print(f"Mean Absolute Error: {mae_rf_values[i]}")
    print()


titles = ['PR']
labels = ['Measured PR', 'Predicted PR']

for i in range(min(y_test.shape[1], len(rf_models))):
    plt.figure(figsize=(8, 8))
    
    actual_data = y_test[:, i]
    predicted_data = rf_models[i].predict(x_test)

    # Check if shapes are compatible for plotting
    if actual_data.shape != predicted_data.shape:
        raise ValueError(f"Shapes of actual_data and predicted_data do not match for target variable {i}")

    # Get the range of the actual data for both axes
    data_range = max(actual_data.max(), predicted_data.max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]}')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()
##SCATTER PLOT OF ACTUAL DATA AGAINST PREDICTED DATA FOR THE ENTIRE DATA THAT IS THE PREDICTION WAS DONE ON THE ENTIRE Y 
    
 
# Lists to store predictions for each target variable
test_rf_all = []

# Iterate over each target variable
for i in range(y.shape[1]):
    # Make predictions for the current target variable
    test_rf = rf_models[i].predict(x)
    test_rf_all.append(test_rf)

# Now, test_y3_all is a 2D NumPy array with three columns, each representing a target variable
test_rf_all = np.column_stack(test_rf_all)

for i in range(min(y.shape[1], test_rf_all.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y[:, i]
    predicted_data = test_rf_all[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y[:, i].max(), test_rf_all[:, i].max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]} for Random forest')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()
    

#Plot the whole measured data against predicted data
plt.figure(figsize = (10, 6))
plt.plot(df1['DEPT'], y[:,0], label='actual PR')
plt.plot(df1['DEPT'], test_rf_all[:,0], label='Predicted PR')

# plt.yscale('log')

# Adjust y-axis limits in terms of logarithmic values
plt.ylim(0.0, 0.6)  # Example limits, adjust as needed

# plt.ylim(0, 3)
plt.title("Actual PR VS Predicted PR Using Random forest")
plt.xlabel('Depth')
plt.ylabel('PR')
plt.legend()
plt.grid()


## XGBOOST ALGORITHM  ##

# Create an empty list to store the trained XGBoost models
xgboost_regressors = []

# Iterate over each column in y_train
for i in range(y_train.shape[1]):
    # Create XGBoost regressor
    xgboost_regressor = XGBRegressor(n_estimators=50, random_state=42)
    # Fit XGBoost regressor to the i-th column of y_train
    xgboost_regressor.fit(x_train, y_train[:, i])
    # Append trained XGBoost regressor to the list
    xgboost_regressors.append(xgboost_regressor)



# Now xgboost_regressors contains the trained XGBoost models with the best hyperparameters

## Using randomizedSearchCV
# Define a dictionary of hyperparameters to search
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'learning_rate': [0.01, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30],
    'max_depth': [1, 3, 5, 7],
    'subsample': [0.5, 0.7, 0.9],
    'colsample_bytree': [0.5, 0.7, 0.9]
    # Add more hyperparameters to search over if needed
}

# Create an empty list to store the trained XGBoost models
xgboost_regressors = []

# Iterate over each column in y_train
for i in range(y_train.shape[1]):
    # Create XGBoost regressor
    xgboost_regressor = XGBRegressor(random_state=42)

    # Perform randomized search for the current target variable
    random_search = RandomizedSearchCV(estimator=xgboost_regressor, param_distributions=param_grid,
                                       n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(x_train, y_train[:, i])

    # Get the best estimator from the randomized search
    best_estimator = random_search.best_estimator_

    # Append the best estimator to the list of trained XGBoost regressors
    xgboost_regressors.append(best_estimator)

    # Print the best parameters and best score for the current target variable
    print(f"Best parameters for target variable {i+1}: {random_search.best_params_}")
    print(f"Best score for target variable {i+1}: {random_search.best_score_}")


    
# Lists to store performance metrics for each target variable
r_xgb_values = []
mse_xgb_values = []
rmse_xgb_values = []
mae_xgb_values = []

# Iterate over each target variable
for i in range(y_test.shape[1]):
    # Make predictions for the current target variable
    y_pred_xgb = np.column_stack([regressor.predict(x_test) for regressor in xgboost_regressors])

    # Compute R2 for the current target variable
    r_xgb = r2_score(y_test[:, i], y_pred_xgb[:, i])
    r_xgb_values.append(r_xgb)

    # Compute MSE for the current target variable
    mse_xgb = mean_squared_error(y_test[:, i], y_pred_xgb[:, i])
    mse_xgb_values.append(mse_xgb)

    # Compute RMSE for the current target variable
    rmse_xgb = mse_xgb**0.5
    rmse_xgb_values.append(rmse_xgb)

    # Compute MAE for the current target variable
    mae_xgb = mean_absolute_error(y_test[:, i], y_pred_xgb[:, i])
    mae_xgb_values.append(mae_xgb)
    
for i in range(len(r_xgb_values)):
    print(f"Metrics for target variable {i+1}:")
    print(f"R2 Score: {r_xgb_values[i]}")
    print(f"Mean Squared Error: {mse_xgb_values[i]}")
    print(f"Root Mean Squared Error: {rmse_xgb_values[i]}")
    print(f"Mean Absolute Error: {mae_xgb_values[i]}")
    print()
    
titles = ['PR']
labels = ['Measured PR', 'Predicted PR']

for i in range(min(y_test.shape[1], len(xgboost_regressors))):
    plt.figure(figsize=(8, 8))
    
    actual_data = y_test[:, i]
    predicted_data = xgboost_regressors[i].predict(x_test)

    # Check if shapes are compatible for plotting
    if actual_data.shape != predicted_data.shape:
        raise ValueError(f"Shapes of actual_data and predicted_data do not match for target variable {i}")

    # Get the range of the actual data for both axes
    data_range = max(actual_data.max(), predicted_data.max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]}')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()
    
 ##SCATTER PLOT OF ACTUAL DATA AGAINST PREDICTED DATA FOR THE ENTIRE DATA THAT IS THE PREDICTION WAS DONE ON THE ENTIRE Y

# Lists to store predictions for each target variable

test_xgb_all = []

# Iterate over each target variable
for i in range(y.shape[1]):
    # Make predictions for the current target variable
    test_xgb = xgboost_regressors[i].predict(x)
    test_xgb_all.append(test_xgb)

# Now, test_y3_all is a 2D NumPy array with three columns, each representing a target variable
test_xgb_all = np.column_stack(test_xgb_all)

for i in range(min(y.shape[1], test_xgb_all.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y[:, i]
    predicted_data = test_xgb_all[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y[:, i].max(), test_xgb_all[:, i].max())
    
    plt.xlim(0, data_range)
    plt.ylim(0, data_range)
    
    # Plot the scatter plot
    plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
    # Plot the ideal line dynamically based on the data range
    plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
    # Customize plot titles and axis labels
    plt.title(f'Scatter Plot - {titles[i]}')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()

#Plot the whole measured data against predicted data    
plt.figure(figsize = (10, 6))
plt.plot(df1['DEPT'], y[:,0], label='actual PR')
plt.plot(df1['DEPT'], test_xgb_all[:,0], label='Predicted PR')

# plt.yscale('log')

# Adjust y-axis limits in terms of logarithmic values
plt.ylim(0.0, 0.6)  # Example limits, adjust as needed

plt.title("Actual PR VS Predicted PR Using XGBOOST Regression")
plt.xlabel('Depth')
plt.ylabel('PR')
plt.legend()
plt.grid()


## Response Surface Methodology
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# # Load the data
# df = pd.read_csv('path_to_your_data.csv')

# Define the independent variables (DTCO and DTSM) and the dependent variable (actual Poisson's ratio)
X = new_df2[['DTCO', 'DTSM']]
Y = new_df2['PR']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Generate polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly_train = poly.fit_transform(X_train)
X_poly_test = poly.transform(X_test)

# Fit a polynomial regression model
model = sm.OLS(Y_train, sm.add_constant(X_poly_train)).fit()

# Extract coefficients
coefficients = model.params
feature_names = poly.get_feature_names_out(X.columns)

# Display the equation
equation = "Poisson's ratio = {:.4f}".format(coefficients[0])
for i, coef in enumerate(coefficients[1:]):
    equation += " + ({:.4f} * {})".format(coef, feature_names[i])

print("Fitted Equation:")
print(equation)

# Predict on the test set
Y_pred = model.predict(sm.add_constant(X_poly_test))

# Calculate RMSE and MAE
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"RMSE: {rmse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R-squared: {r2:.4f}")

# Print the model summary
print(model.summary())

# Plot the calculated Poisson's ratio vs actual Poisson's ratio
plt.figure(figsize=(10, 6))
plt.plot(new_df2['DEPT'], Y, 'r--', label='Actual Poisson ratio')
plt.plot(new_df2['DEPT'], model.predict(sm.add_constant(poly.transform(X))), label='Calculated Poisson ratio using RSM')
plt.ylim(0, 3.0)
plt.title("RSM Calculated Poisson's ratio VS Actual Poisson's ratio Using RSM")
plt.xlabel('Depth')
plt.ylabel("Poisson's ratio")
plt.legend()
plt.grid()
plt.show()
