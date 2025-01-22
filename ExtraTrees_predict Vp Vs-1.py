# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 10:31:51 2024

@author: yvann
"""





x = new_df1.iloc[:,0:7].values #independent variables
y = new_df1.iloc[:,7:].values  #dependent variable

#Split training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

#fitting Random forest Regression to the training set 
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV


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


titles = ['DTCO', 'DTSM']
labels = ['Actual DTCO', 'Predicted DTCO', 'Actual DTSM', 'Predicted DTSM']

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
    plt.title(f'Scatter Plot - {titles[i]}')
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
    plt.title(f'Scatter Plot - {titles[i]}')
    plt.xlabel(f'{labels[2*i]}')
    plt.ylabel(f'{labels[2*i+1]}')
    
    plt.legend()
    plt.show()


#Plot the whole measured data against predicted data
plt.figure(figsize = (15, 5))
plt.plot(df1['DEPT'], y[:,0], label='actual DTCO')
plt.plot(df1['DEPT'], test_ExtraT_all[:,0], label='Predicted DTCO')

# plt.yscale('log')

# Adjust y-axis limits in terms of logarithmic values
plt.ylim(0.0, 380)  # Example limits, adjust as needed

# plt.ylim(0, 3)
plt.title("Actual DTCO VS Predicted DTCO Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('DTCO')
plt.legend()
plt.grid()



plt.figure(figsize = (15, 5))
plt.plot(df1['DEPT'], y[:,1], label='actual DTSM')
plt.plot(df1['DEPT'], test_ExtraT_all[:,1], label='Predicted DTSM')
plt.ylim(0, 380)
plt.title("Actual DTSM VS Predicted DTSM Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('DTSM')
plt.legend()
plt.grid()




import numpy as np
import matplotlib.pyplot as plt

# Data
algorithms = ['Decision Tree', 'Random Forest', 'XGBoost', 'Extreme Tree']
r_squared_values = [98, 98, 97, 98]
rmse_values = [3.7, 3.0, 4.0, 3.3]
mae_values = [1.5, 1.2, 1.7, 1.2]

# Group data by algorithm for easier plotting
metrics = ['R-squared', 'RMSE', 'MAE']
data = np.array([r_squared_values, rmse_values, mae_values])

# Plotting
x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each metric
for i, metric in enumerate(metrics):
    bars = ax.bar(x + i*width, data[i], width, label=metric)
    # Add scores on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Algorithms')
ax.set_ylabel('Metric scores')
ax.set_title('Comparison of R-squared, RMSE, and MAE for Different Algorithms in DTCO prediction')
ax.set_xticks(x + width)
ax.set_xticklabels(algorithms)
ax.legend()

fig.tight_layout()

plt.show()

## Bar chart display
# Data
algorithms = ['Decision Tree', 'Random Forest', 'XGBoost', 'Extreme Tree']
r_squared_values = [97, 98, 98, 99]
rmse_values = [11.4, 7.7, 10.1, 7.4]
mae_values = [4.9, 3.8, 5.5, 3.8]

# Group data by algorithm for easier plotting
metrics = ['R-squared', 'RMSE', 'MAE']
data = np.array([r_squared_values, rmse_values, mae_values])

# Plotting
x = np.arange(len(algorithms))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each metric
for i, metric in enumerate(metrics):
    bars = ax.bar(x + i*width, data[i], width, label=metric)
    # Add scores on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height}', ha='center', va='bottom')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Algorithms')
ax.set_ylabel('Metric scores')
ax.set_title('Comparison of R-squared, RMSE, and MAE for Different Algorithms in DTSM prediction')
ax.set_xticks(x + width)
ax.set_xticklabels(algorithms)
ax.legend()

fig.tight_layout()

plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Assuming df1 is your DataFrame containing DTCO and DTSM
# Example data
df1 = pd.read_csv('path_to_your_data.csv')

# Filter out any negative or invalid values if necessary
df1 = df1[(df1['DTCO'] > 0) & (df1['DTSM'] > 0)]

# Extract the relevant columns
dtco1 = new_df1['DTCO'].values.reshape(-1, 1)
dtsm1 = new_df1['DTSM'].values.reshape(-1, 1)

# Perform linear regression
regression_model = LinearRegression()
regression_model.fit(dtco1, dtsm1)

# Get the slope (m) and intercept (b) of the line
slope = regression_model.coef_[0][0]
intercept = regression_model.intercept_[0]

# Predict the DTSM values based on the DTCO values
dtsm1_pred = regression_model.predict(dtco1)

# Calculate R-squared value
r_squared = r2_score(dtsm1, dtsm1_pred)

# Print the equation of the line and R-squared value
print(f"Correlation Equation: DTSM = {slope:.4f} * DTCO + {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(dtco1, dtsm1, label='Data Points', color='blue')
plt.plot(dtco1, dtsm1_pred, label=f'Fit Line: DTSM = {slope:.4f} * DTCO + {intercept:.4f}\nR-squared: {r_squared:.4f}', color='red')
plt.xlabel('DTCO (Compressional Wave Velocity)')
plt.ylabel('DTSM (Shear Wave Velocity)')
plt.title('Cross-plot of DTCO vs DTSM')
plt.legend()
plt.grid(True)
plt.show()

## Apply RSM to my data 

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
## Linear model
# Load your dataset
df1 = pd.read_csv('path_to_your_data.csv')

# Filter out any negative or invalid values
df1 = df1[(df1['DTCO'] > 0) & (df1['DTSM'] > 0)]

# Define the response and predictors
response = df1['DTSM']
predictor = df1['DTCO']

# Add a constant term for the intercept
predictor = sm.add_constant(predictor)

# Fit a linear regression model using OLS (Ordinary Least Squares)
linear_model = sm.OLS(response, predictor).fit()

# Print the model summary
print(linear_model.summary())

# Extract the coefficients
intercept = linear_model.params[0]
slope = linear_model.params[1]

# Print the regression equation
print(f"Linear Model Equation: DTSM = {slope:.4f} * DTCO + {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Predict the values
linear_predictions = linear_model.predict(predictor)

# Calculate RMSE and MAE
linear_rmse = mean_squared_error(response, linear_predictions, squared=False)
linear_mae = mean_absolute_error(response, linear_predictions)

print(f"Linear Model RMSE: {linear_rmse:.4f}")
print(f"Linear Model MAE: {linear_mae:.4f}")

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(dtco1, dtsm1, label='Data Points', color='blue')
plt.plot(dtco1, dtsm1_pred, label=f'Fit Line: DTSM = {slope:.4f} * DTCO + {intercept:.4f}\nR-squared: {r_squared:.4f}', color='red')
plt.xlabel('DTCO (Compressional Wave Velocity)')
plt.ylabel('DTSM (Shear Wave Velocity)')
plt.title('Cross-plot of DTCO vs DTSM')
plt.legend()
plt.grid(True)
plt.show()


## QUADRATIC MODEL
# Add quadratic term
df1['DTCO_squared'] = df1['DTCO']**2

# Define the response and predictors
predictors = df1[['DTCO', 'DTCO_squared']]
response = df1['DTSM']

# Add a constant term for the intercept
predictors = sm.add_constant(predictors)

# Fit a quadratic regression model using OLS
quadratic_model = sm.OLS(response, predictors).fit()

# Print the model summary
print(quadratic_model.summary())

# Extract the coefficients
intercept = quadratic_model.params[0]
slope = quadratic_model.params[1]
quadratic_term = quadratic_model.params[2]

# Print the regression equation
print(f"Quadratic Model Equation: DTSM = {quadratic_term:.4f} * DTCO^2 + {slope:.4f} * DTCO + {intercept:.4f}")
print(f"R-squared: {r_squared:.4f}")

# Predict the values
quadratic_predictions = quadratic_model.predict(predictors)

# Calculate RMSE and MAE
quadratic_rmse = mean_squared_error(response, quadratic_predictions, squared=False)
quadratic_mae = mean_absolute_error(response, quadratic_predictions)

print(f"Quadratic Model RMSE: {quadratic_rmse:.4f}")
print(f"Quadratic Model MAE: {quadratic_mae:.4f}")

# Plot the data points and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(dtco1, dtsm1, label='Data Points', color='blue')
plt.plot(dtco_values, quadratic_predictions, label=f'Fit Line: DTSM = {quadratic_term:.4f} * DTCO^2 + {slope:.4f} * DTCO + {intercept:.4f}\nR-squared: {r_squared:.4f}', color='red')
plt.xlabel('DTCO (Compressional Wave Velocity)')
plt.ylabel('DTSM (Shear Wave Velocity)')
plt.title('Cross-plot of DTCO vs DTSM')
plt.legend()
plt.grid(True)
plt.show()
### END RSM

import matplotlib.pyplot as plt

# Assuming df1, test_ExtraT_all, and your empirical formula are already defined

# Define the empirical formula for calculating DTSM from DTCO
def calculate_dtsm_from_dtco(dtco):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for sand beds Vs = 0.7696Vp - 0.86735
    a = 0.7696  # example coefficient
    b = 0.86735   # example intercept
    dtsm = a * dtco - b
    return dtsm

# Extract depth and predicted DTCO data
depth_data = df1['DEPT']
predicted_dtco_data = test_ExtraT_all[:, 0]  # Assuming predicted DTCO is in the first column

# Calculate DTSM using the predicted DTCO values
calculated_dtsm_data = calculate_dtsm_from_dtco(predicted_dtco_data)

# Extract predicted DTSM data
predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column

# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(15, 5))
plt.plot(depth_data, calculated_dtsm_data, label='Calculated DTSM')
plt.plot(depth_data, predicted_dtsm_data, label='Predicted DTSM')
plt.ylim(0, 450)
plt.title("Calculated DTSM VS Predicted DTSM Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('DTSM')
plt.legend()
plt.grid()
plt.show()

# Define the empirical formula for calculating DTSM from DTCO
def calculate_dtsm_from_dtco_for_sandbeds(dtco):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for shale beds Vs = 0.7696Vp - 0.86735
    a = 2.5353  # example coefficient
    b = 35.0291   # example intercept
    dtsm = a * dtco - b
    return dtsm

# Extract depth and predicted DTCO data
depth_data = df1['DEPT']
predicted_dtco_data = test_ExtraT_all[:, 0]  # Assuming predicted DTCO is in the first column

# Calculate DTSM using the predicted DTCO values
calculated_dtsm_data = calculate_dtsm_from_dtco_for_sandbeds(predicted_dtco_data)

# Extract predicted DTSM data
predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column
actual_dtsm_data = y[:, 1]
# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(15, 5))
plt.plot(depth_data, calculated_dtsm_data, label='Calculated DTSM')
# plt.plot(depth_data, predicted_dtsm_data, label='Predicted DTSM')
plt.plot(depth_data, actual_dtsm_data, label='Actual DTSM')
plt.ylim(0, 450)
plt.title("Calculated DTSM VS Actual DTSM")
plt.xlabel('Depth')
plt.ylabel('DTSM')
plt.legend()
plt.grid()
plt.show()


# Define the empirical formula for calculating DTSM from DTCO
def calculate_dtsm_from_Castagna(dtco):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for shale beds Vs = 0.7696Vp - 0.86735
    a = -0.05509  # example coefficient
    b = 1.0168   # example intercept
    c = 1.0305
    dtsm = a * dtco ** 2 + b * dtco - c
    return dtsm

# Extract depth and predicted DTCO data
depth_data = df1['DEPT']
predicted_dtco_data = test_ExtraT_all[:, 0]  # Assuming predicted DTCO is in the first column

# Calculate DTSM using the predicted DTCO values
calculated_dtsm_data = calculate_dtsm_from_Castagna(predicted_dtco_data)

# Extract predicted DTSM data
predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column

# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(15, 5))
plt.plot(depth_data, calculated_dtsm_data, label='Calculated DTSM')
plt.plot(depth_data, predicted_dtsm_data, label='Predicted DTSM')
plt.ylim(-1000, 450)
plt.title("Castagna Calculated DTSM VS Predicted DTSM Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('DTSM')
plt.legend()
plt.grid()
plt.show()

print(predicted_dtco_data)
print(calculated_dtsm_data)
print(predicted_dtsm_data)


# Define the empirical formula for calculating DTSM from DTCO
def calculate_dtsm_from_Brocher(dtco):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for shale beds Vs = 0.7696Vp - 0.86735
    a = 0.7858  # example coefficient
    b = 1.2344   # example intercept
    c = 0.7949
    d = 0.1238
    e = 0.006
    dtsm = a - (b*dtco) + (c * dtco**2) - (d*dtco**3) + (e*dtco**4)
    return dtsm

# Extract depth and predicted DTCO data
depth_data = df1['DEPT']
predicted_dtco_data = test_ExtraT_all[:, 0]  # Assuming predicted DTCO is in the first column

# Calculate DTSM using the predicted DTCO values
calculated_dtsm_data = calculate_dtsm_from_Castagna(predicted_dtco_data)

# Extract predicted DTSM data
predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column

# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(15, 5))
plt.plot(depth_data, calculated_dtsm_data, label='Calculated DTSM')
plt.plot(depth_data, predicted_dtsm_data, label='Predicted DTSM')
plt.ylim(-1000, 450)
plt.title("Brocher Calculated DTSM VS Predicted DTSM Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('DTSM')
plt.legend()
plt.grid()
plt.show()

def calculate_dtsm_from_carroll(dtco):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for shale beds Vs = 0.7696Vp - 0.86735
    a = 1.09913326  # example coefficient
    b = 0.9238115336   # example intercept
    dtsm = a * dtco**b
    return dtsm

# Extract depth and predicted DTCO data
depth_data = df1['DEPT']
predicted_dtco_data = test_ExtraT_all[:, 0]  # Assuming predicted DTCO is in the first column

# Calculate DTSM using the predicted DTCO values
calculated_dtsm_data = calculate_dtsm_from_Castagna(predicted_dtco_data)

# Extract predicted DTSM data
predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column

# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(15, 5))
plt.plot(depth_data, calculated_dtsm_data, label='Calculated DTSM')
plt.plot(depth_data, predicted_dtsm_data, label='Predicted DTSM')
plt.ylim(-1000, 450)
plt.title("Carroll Calculated DTSM VS Predicted DTSM Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('DTSM')
plt.legend()
plt.grid()
plt.show()
# Extract DTSM and DTCO columns
# DTSM = new_df1['DTSM']  # Replace 'DTSM' with the actual column name for DTSM in your dataset
# DTCO = new_df1['DTCO']  # Replace 'DTCO' with the actual column name for DTCO in your dataset

# # Calculate the Pearson correlation coefficient
# correlation = DTSM.corr(DTCO)
# print(f'Pearson correlation coefficient between DTSM and DTCO: {correlation:.4f}')

# # Create a scatter plot (crossplot)
# plt.figure(figsize=(10, 6))
# plt.scatter(DTCO, DTSM, alpha=0.5)
# plt.title('Crossplot of DTSM vs. DTCO')
# plt.xlabel('DTCO')
# plt.ylabel('DTSM')
# plt.grid(True)
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Extract the actual DTSM data from your dataset
actual_dtsm_data = y[:, 1]  # Assuming actual DTSM is in the second column of y

# Extract the predicted DTSM data from your model output
predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column of test_ExtraT_all

# Calculate regression metrics
mae = mean_absolute_error(actual_dtsm_data, predicted_dtsm_data)
mse = mean_squared_error(actual_dtsm_data, predicted_dtsm_data)
rmse = np.sqrt(mse)
r2 = r2_score(actual_dtsm_data, predicted_dtsm_data)

# Create the crossplot (scatter plot of actual vs. predicted DTSM)
plt.figure(figsize=(10, 6))
plt.scatter(actual_dtsm_data, predicted_dtsm_data, alpha=0.5, label='Data Points')

# Add a 1:1 line for reference
min_val = min(min(actual_dtsm_data), min(predicted_dtsm_data))
max_val = max(max(actual_dtsm_data), max(predicted_dtsm_data))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal 1:1 Line')

# Add labels and title
plt.xlabel('Actual DTSM')
plt.ylabel('Predicted DTSM')
plt.title('Crossplot of Actual vs. Predicted DTSM')
plt.legend()
plt.grid(True)

# Display regression metrics on the plot
metrics_text = f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}'
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Show the plot
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Extract the actual DTSM data from your dataset
actual_dtsm_data = y[:, 1]  # Assuming actual DTSM is in the second column of y

# Extract the predicted DTSM data from your model output
# predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column of test_ExtraT_all

# Calculate regression metrics
mae = mean_absolute_error(actual_dtsm_data, calculated_dtsm_data)
mse = mean_squared_error(actual_dtsm_data, calculated_dtsm_data)
rmse = np.sqrt(mse)
r2 = r2_score(actual_dtsm_data, calculated_dtsm_data)

# Create the crossplot (scatter plot of actual vs. predicted DTSM)
plt.figure(figsize=(10, 6))
plt.scatter(actual_dtsm_data, calculated_dtsm_data, alpha=0.5, label='Data Points')

# Add a 1:1 line for reference
min_val = min(min(actual_dtsm_data), min(calculated_dtsm_data))
max_val = max(max(actual_dtsm_data), max(calculated_dtsm_data))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal 1:1 Line')

# Add labels and title
plt.xlabel('Actual DTSM')
plt.ylabel('Calculated DTSM')
plt.title('Crossplot of Actual vs. Calculated DTSM')
# Move the legend to the upper left outside the plot
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Display regression metrics on the plot
metrics_text = f'MAE: {mae:.2f}\nMSE: {mse:.2f}\nRMSE: {rmse:.2f}\nR²: {r2:.2f}'
plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

# Show the plot
plt.tight_layout()  # Adjust layout to make room for the legend
plt.grid(True)
plt.show()


# Define the empirical formula for calculating poisson's ratio
def calculate_poisson_ratio(DTCO, DTSM):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for dynamic poisson's ratio
   
    Poisson_ratio = ((DTCO**2) - 2*(DTSM**2)) / (2 * ((DTCO**2) - (DTSM**2)))
    return Poisson_ratio

# Extract depth and predicted DTCO data
depth_data = df1['DEPT']
# predicted_dtco_data = test_ExtraT_all[:, 0]  # Assuming predicted DTCO is in the first column
actual_dtco_data = x[:, 6]
# Extract predicted DTSM data
# predicted_dtsm_data = test_ExtraT_all[:, 1]  # Assuming predicted DTSM is in the second column
actual_dtsm_data = x[:, 7]
# Calculate DTSM using the predicted DTCO values
calculated_poisson_ratios = calculate_poisson_ratio(actual_dtco_data, actual_dtsm_data)

Actual_poisson_ratio = new_df2.iloc[:, 8]

# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(10, 6))
plt.plot(depth_data, calculated_poisson_ratios, label='Calculated poissons ratio')
plt.plot(depth_data, Actual_poisson_ratio, 'r--', label='Actual poissons ratio')
plt.ylim(0, 3.0)
plt.title("Calculated poissons ratio VS Actual poissons ratio Using Extreme Tree")
plt.xlabel('Depth')
plt.ylabel('poissons ratio')
plt.legend()
plt.grid()
plt.show()

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# # Example data
# # Replace these with your actual data
# depth_data = np.linspace(0, 12000, num=12001)  # Example depth data
# actual_dtco_data = np.random.uniform(0.1, 0.4, size=12001)  # Replace with your actual DTCO data
# actual_dtsm_data = np.random.uniform(0.1, 0.4, size=12001)  # Replace with your actual DTSM data

# Initial empirical formula for Poisson's ratio
def empirical_poisson_ratio(DTCO, DTSM):
    return ((DTCO**2) - 2*(DTSM**2)) / (2 * ((DTCO**2) - (DTSM**2)))

# RSM formula for Poisson's ratio
def rsm_poisson_ratio(DTCO, DTSM):
    return 0.3389 - 0.0090 * DTCO + 0.0042 * DTSM

# Calculate Poisson's ratios using both formulas
empirical_poisson_ratios = empirical_poisson_ratio(actual_dtco_data, actual_dtsm_data)
rsm_poisson_ratios = rsm_poisson_ratio(actual_dtco_data, actual_dtsm_data)

# Calculate MAE
mae = mean_absolute_error(empirical_poisson_ratios, rsm_poisson_ratios)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(empirical_poisson_ratios, rsm_poisson_ratios))

# Print the results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# Define the empirical formula for calculating Shear Modulus
def calculate_Shear_Modulus(RHOZ, DTSM):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for SHear Modulus
   
    Shear_Modulus = RHOZ * (DTSM**2) 
    return Shear_Modulus

# Extract depth
depth_data = new_df1['DEPT']

Density_data = new_df1.iloc[:, 5]
# Extract predicted DTSM data
predicted_dtsm_data_ET = test_ExtraT_all[:, 1]

predicted_dtsm_data_RF = test_rf_all[:, 1]

Actual_dtsm = new_df1.iloc[:, 7]

# Calculate DTSM using the predicted values
Calculated_Shear_moduli_ET = calculate_Shear_Modulus(Density_data, predicted_dtsm_data_ET)
Calculated_Shear_moduli_RF = calculate_Shear_Modulus(Density_data, predicted_dtsm_data_RF)
Calculated_Shear_moduli_Actual_dtsm = calculate_Shear_Modulus(Density_data, Actual_dtsm)

# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(10, 6))
plt.plot(depth_data, Calculated_Shear_moduli_ET, label='Calculated Shear Modulus using predicted Extreme tree Vs')
plt.plot(depth_data, Calculated_Shear_moduli_RF, 'r--', label='Calculated Shear Modulus using predicted Random forest Vs')
plt.plot(depth_data, Calculated_Shear_moduli_Actual_dtsm,  label='Calculated Shear Modulus using Actual_dtsm')
plt.ylim(0, 500000)
plt.title("Calculated Shear Modulus using predicted Extreme tree VS Calculated Shear Modulus using predicted Random forest Vs")
plt.xlabel('Depth')
plt.ylabel('Shear Modulus')
plt.legend()
plt.grid()
plt.show()

def calculate_Bulk_Modulus(RHOZ, DTCO, DTSM):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for SHear Modulus
   
    Bulk_Modulus = (RHOZ * (DTCO**2)) - ((4/3) * (RHOZ * (DTSM**2))) 
    return Bulk_Modulus

# Extract depth
depth_data = new_df1['DEPT']

Density_data = new_df1.iloc[:, 5]
# Extract predicted DTSM data
predicted_dtsm_data_ET = test_ExtraT_all[:, 1]

predicted_dtsm_data_RF = test_rf_all[:, 1]

predicted_dtco_data_ET = test_ExtraT_all[:, 0]

predicted_dtco_data_RF = test_rf_all[:, 0]

Actual_dtsm = new_df1.iloc[:, 7]
Actual_dtco = new_df1.iloc[:, 6]

# Calculate DTSM using the predicted values
Calculated_Bulk_moduli_ET = calculate_Bulk_Modulus(Density_data, predicted_dtco_data_ET, predicted_dtsm_data_ET)
Calculated_Bulk_moduli_RF = calculate_Bulk_Modulus(Density_data, predicted_dtco_data_RF, predicted_dtsm_data_RF)
Calculated_Bulk_moduli_Actual = calculate_Bulk_Modulus(Density_data, Actual_dtco , Actual_dtsm)


# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(10, 6))
plt.plot(depth_data, Calculated_Bulk_moduli_ET, label='Calculated Bulk Modulus using predicted Extreme tree Vp Vs', linestyle='-', alpha=0.7)
plt.plot(depth_data, Calculated_Bulk_moduli_RF, 'r--', label='Calculated Bulk Modulus using predicted Random forest Vp Vs', alpha=0.7)
plt.plot(depth_data, Calculated_Bulk_moduli_Actual, label='Calculated Bulk Modulus using Actual Vp Vs', linestyle=':', alpha=0.7)
plt.ylim(-1000000, 10000)
plt.title("Calculated Bulk Modulus using predicted Extreme tree, Random forest, and actual shear velocity Vs")
plt.xlabel('Depth')
plt.ylabel('Bulk Modulus')
plt.legend()
plt.grid()
plt.show()

print(Calculated_Bulk_moduli_Actual)


def calculate_Youngs_modulus(RHOZ, DTCO, DTSM):
    # Replace this with the actual empirical formula you have
    # I used empirical formula for youngs Modulus
   
    Youngs_modulus = (RHOZ * (DTSM**2)) * (((3*(DTCO**2)) -(4*(DTSM**2)))/((DTCO**2) - (DTSM**2))) 
    return Youngs_modulus

# Extract depth
depth_data = new_df1['DEPT']

Density_data = new_df1.iloc[:, 5]
# Extract predicted DTSM data
predicted_dtsm_data_ET = test_ExtraT_all[:, 1]

predicted_dtsm_data_RF = test_rf_all[:, 1]

predicted_dtco_data_ET = test_ExtraT_all[:, 0]

predicted_dtco_data_RF = test_rf_all[:, 0]

Actual_dtsm = new_df1.iloc[:, 7]
Actual_dtco = new_df1.iloc[:, 6]

# Calculate DTSM using the predicted values
Calculated_Youngs_modulus_ET = calculate_Youngs_modulus(Density_data, predicted_dtco_data_ET, predicted_dtsm_data_ET)
Calculated_Youngs_modulus_RF = calculate_Youngs_modulus(Density_data, predicted_dtco_data_RF, predicted_dtsm_data_RF)

# Calculate DTSM using the actual values
Calculated_Youngs_modulus_Actual = calculate_Youngs_modulus(Density_data, Actual_dtco , Actual_dtsm)


# Plot the calculated DTSM vs predicted DTSM
plt.figure(figsize=(10, 6))
plt.plot(depth_data, Calculated_Youngs_modulus_ET, label='Calculated Youngs Modulus using predicted Extreme tree Vp Vs', linestyle='-', alpha=0.7)
plt.plot(depth_data, Calculated_Youngs_modulus_RF, 'r--', label='Calculated Youngs Modulus using predicted Random forest Vp Vs', alpha=0.7)
plt.plot(depth_data, Calculated_Youngs_modulus_Actual, label='Calculated Youngs Modulus using Actual Vp Vs', linestyle=':', alpha=0.7, color='g')
plt.ylim(0, 10000000)
plt.title("Calculated Youngs Modulus using predicted Extreme tree, Random forest, and actual shear velocity Vs")
plt.xlabel('Depth')
plt.ylabel('Youngs Modulus')
plt.legend()
plt.grid()
plt.show()

