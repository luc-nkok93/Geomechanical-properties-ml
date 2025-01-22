# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:27:58 2024

@author: yvann
"""

# install the missingno lib to use for identifying missing data: !pip install missingno
!pip install missingno
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import missingno as msno
import seaborn as sns


df = pd.read_csv(r"C:\Users\yvann\OneDrive - North Dakota University System\Application of ML in Petrophysics\20457_HOVDEN_Data_for_Geomech.csv")

df.describe()  # description of the data

# Get information about data, non-null and nulls in data
df.info()

df.dtypes

# df = pd.to_numeric(df.replace('[^\d.-]', '', regex=True), errors='coerce')
# df = df.applymap(lambda x: np.nan if isinstance(x, str) and '-999.25' in x else x)
df = df.drop(['DTST', 'NPOR'], axis = 1)
df.replace(-999.25, np.nan, inplace=True)
# number of missing values in the data
df.isna().sum()
df1 = df.dropna(how='any')

df1 = df1[(df1 > 0).all(axis=1)]    
df1.describe()
df1 = df1.drop(['AT10'], axis = 1)

# import seaborn library and plot a heatnap to analyse how data correlates
import seaborn as sns
plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
###df.corr()
#sns.heatmap(df.corr(), annot=True)

# df1 = df1.drop(['PR'], axis = 1)
              
#drop missing data rows

df1.shape
msno.bar(df)
#matrix Chart
msno.matrix(df)

#fill missing data with interpolated vlues
# df2 = df.interpolate(method='linear')
# df2 = df2.fillna(method='bfill')
# df2.isna().sum()

# #fill missing data with mean
# df3 = df.fillna(value=df.mean())
# df3.isna().sum()

sns.boxplot(x='AT10', data=df2)
df2.describe()

# IDENTIFY AND REMOVE OUTLIERS

#IQR method 

# Q1 = df1['AT10'].quantile(0.25)
# Q3 = df1['AT10'].quantile(0.75)
# IQR = Q3 - Q1
# Q1, Q3, IQR

# upper_limit = Q3 + (1.5 * IQR)
# lower_limit = Q1 - (1.5 * IQR)
# lower_limit, upper_limit

# sns.boxplot(x='AT10', data=df1)

# df1.loc[(df1['AT10'] > upper_limit) | (df1['AT10'] < lower_limit)]  #find outliers

# #trim the data
# new_df1 = df1.loc[(df['AT10'] < upper_limit) & (df['AT10'] > lower_limit)]
# print('before removing outliers:', len(df1))
# print('after removing outliers:', len(new_df1))
# print('outliers:', len(df1)-len(new_df1))

# sns.boxplot(x='AT10', data=new_df1)

# #Capping -  change the outlier values to upper (or) lower limit values
# new_df1 = df1.copy()
# new_df1.loc[(new_df1['AT10'] > upper_limit), 'AT10'] = upper_limit
# new_df1.loc[(new_df1['AT10'] < lower_limit), 'AT10'] = lower_limit

# sns.boxplot(x='AT10', data=new_df1)

##Next column
Q1 = df1['AT90'].quantile(0.25)
Q3 = df1['AT90'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='AT90', data=df1)
df1.loc[(df1['AT90'] > upper_limit) | (df1['AT90'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['AT90'] < upper_limit) & (df['AT90'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='AT90', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['AT90'] > upper_limit), 'AT90'] = upper_limit
new_df1.loc[(new_df1['AT90'] < lower_limit), 'AT90'] = lower_limit

sns.boxplot(x='AT90', data=new_df1)

#next Column
Q1 = df1['DTCO'].quantile(0.25)
Q3 = df1['DTCO'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='DTCO', data=df1)
df1.loc[(df1['DTCO'] > upper_limit) | (df1['DTCO'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['DTCO'] < upper_limit) & (df['DTCO'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='DTCO', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['DTCO'] > upper_limit), 'DTCO'] = upper_limit
new_df1.loc[(new_df1['DTCO'] < lower_limit), 'DTCO'] = lower_limit

sns.boxplot(x='DTCO', data=new_df1)

##Next column
Q1 = df1['GR'].quantile(0.25)
Q3 = df1['GR'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='GR', data=df1)
df1.loc[(df1['GR'] > upper_limit) | (df1['GR'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['GR'] < upper_limit) & (df['GR'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='GR', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['GR'] > upper_limit), 'GR'] = upper_limit
new_df1.loc[(new_df1['GR'] < lower_limit), 'GR'] = lower_limit

sns.boxplot(x='GR', data=new_df1)

#next Column
Q1 = df1['HCAL'].quantile(0.25)
Q3 = df1['HCAL'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='HCAL', data=df1)
df1.loc[(df1['HCAL'] > upper_limit) | (df1['HCAL'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['HCAL'] < upper_limit) & (df['HCAL'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='HCAL', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['HCAL'] > upper_limit), 'HCAL'] = upper_limit
new_df1.loc[(new_df1['HCAL'] < lower_limit), 'HCAL'] = lower_limit

sns.boxplot(x='HCAL', data=new_df1)


#next Column
Q1 = df1['NPHI'].quantile(0.25)
Q3 = df1['NPHI'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='NPHI', data=df1)
df1.loc[(df1['NPHI'] > upper_limit) | (df1['NPHI'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['NPHI'] < upper_limit) & (df['NPHI'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='NPHI', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['NPHI'] > upper_limit), 'NPHI'] = upper_limit
new_df1.loc[(new_df1['NPHI'] < lower_limit), 'NPHI'] = lower_limit

sns.boxplot(x='NPHI', data=new_df1)

#next Column
Q1 = df1['DTSM'].quantile(0.25)
Q3 = df1['DTSM'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='DTSM', data=df1)
df1.loc[(df1['DTSM'] > upper_limit) | (df1['DTSM'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['DTSM'] < upper_limit) & (df['DTSM'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='DTSM', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['DTSM'] > upper_limit), 'DTSM'] = upper_limit
new_df1.loc[(new_df1['DTSM'] < lower_limit), 'DTSM'] = lower_limit

sns.boxplot(x='DTSM', data=new_df1)

#next Column
Q1 = df1['RHOZ'].quantile(0.25)
Q3 = df1['RHOZ'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='RHOZ', data=df1)
df1.loc[(df1['RHOZ'] > upper_limit) | (df1['RHOZ'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['RHOZ'] < upper_limit) & (df['RHOZ'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='RHOZ', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['RHOZ'] > upper_limit), 'RHOZ'] = upper_limit
new_df1.loc[(new_df1['RHOZ'] < lower_limit), 'RHOZ'] = lower_limit

sns.boxplot(x='RHOZ', data=new_df1)

#next Column
Q1 = df1['PR'].quantile(0.25)
Q3 = df1['PR'].quantile(0.75)
IQR = Q3 - Q1
Q1, Q3, IQR
upper_limit = Q3 + (1.5 * IQR)
lower_limit = Q1 - (1.5 * IQR)
lower_limit, upper_limit

sns.boxplot(x='PR', data=df1)
df1.loc[(df1['PR'] > upper_limit) | (df1['PR'] < lower_limit)]
#find outliers

#trim the data
new_df1 = df1.loc[(df['PR'] < upper_limit) & (df['PR'] > lower_limit)]
print('before removing outliers:', len(df1))
print('after removing outliers:', len(new_df1))
print('outliers:', len(df1)-len(new_df1))

sns.boxplot(x='PR', data=new_df1)

#Capping -  change the outlier values to upper (or) lower limit values
new_df1 = df1.copy()
new_df1.loc[(new_df1['PR'] > upper_limit), 'PR'] = upper_limit
new_df1.loc[(new_df1['PR'] < lower_limit), 'PR'] = lower_limit

sns.boxplot(x='PR', data=new_df1)

new_df2 = new_df1.copy()
new_df1 = new_df1.drop(['PR'], axis = 1) # dropped PR


plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(new_df1.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);



cols_to_plot = ['DEPT', 'AT90', 'GR', 'HCAL', 'NPHI', 'RHOZ', 'DTCO', 'DTSM']
sns.pairplot(new_df1[cols_to_plot], diag_kind='kde')  ## pairplot for this data computationally expensive?

x = new_df1.iloc[:,0:6].values #independent variables
y = new_df1.iloc[:,6:].values  #dependent variable


#Split training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


### Here preprocessing finish ####



####    Modeling   #####



#Fitting Decision Tree regression to the training set 
from sklearn import metrics
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV


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

## Using randomizedSearchCV
# Define a dictionary of hyperparameters to search
# Define a dictionary of hyperparameters to search
param_grid = {
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
    # Add more hyperparameters to search over if needed
}

# Create an empty list to store the trained DTR models
Dtr_models = []

# Iterate over each column in y_train
for i in range(y_train.shape[1]):
    # Create DecisionTree model
    Dtr = DecisionTreeRegressor(random_state=0)
    
    # Perform randomized search for the current target variable
    random_search = RandomizedSearchCV(estimator=Dtr, param_distributions=param_grid,
                                       n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    random_search.fit(x_train, y_train[:, i])
    
    # Get the best estimator from the randomized search
    best_estimator = random_search.best_estimator_
    
    # Append the best estimator to the list of trained Decision Tree regressors
    Dtr_models.append(best_estimator)
    
    # Print the best parameters and best score for the current target variable
    print(f"Best parameters for target variable {i+1}: {random_search.best_params_}")
    print(f"Best score for target variable {i+1}: {random_search.best_score_}")

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


titles = ['DTCO', 'DTSM']
labels = ['Actual DTCO', 'Predicted DTCO', 'Actual DTSM', 'Predicted DTSM']

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
plt.plot(df1['DEPT'], y[:,0], label='actual DTCO')
plt.plot(df1['DEPT'], test_Dtr_all[:,0], label='Predicted DTCO')

# Adjust y-axis limits in terms of logarithmic values
plt.ylim(0.0, 380)  # Example limits, adjust as needed

# plt.ylim(0, 3)
plt.title("Actual DTCO VS Predicted DTCO Using DecisionTree Regression")
plt.xlabel('Depth')
plt.ylabel('DTCO')
plt.legend()
plt.grid()



plt.figure(figsize = (15, 5))
plt.plot(df1['DEPT'], y[:,1], label='actual DTSM')
plt.plot(df1['DEPT'], test_Dtr_all[:,1], label='Predicted DTSM')
plt.ylim(0, 380)
plt.title("Actual DTSM VS Predicted DTSM Using DecisionTree Regression")
plt.xlabel('Depth')
plt.ylabel('DTSM')
plt.legend()
plt.grid()

