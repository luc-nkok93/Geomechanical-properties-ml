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
new_df1 = new_df1.drop(['PR'], axis = 1) # dropped column HTEM and SP


plt.figure(figsize=(10, 6))
heatmap = sns.heatmap(new_df1.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);
###df.corr()
#sns.heatmap(df.corr(), annot=True)
new_df2.corr()["Perm Air"].sort_values(ascending=False) #sort the correlation coefficient values
new_df2.corr()["Porosity"].sort_values(ascending=False)
new_df2.corr()["water Saturation, percent"].sort_values(ascending=False)

# new_df2.rename(columns={'Perm Air':'Permeability'}, inplace=True) # rename a column


# train_features = pd.DataFrame(data = new_df2)
# sns.pairplot(train_features) 
# sns.pairplot(new_df2) 

cols_to_plot = ['DEPT', 'AT10', 'AT90', 'GR', 'HCAL', 'NPHI', 'RHOZ', 'DTCO', 'DTSM']
sns.pairplot(new_df1[cols_to_plot], diag_kind='kde')  ## pairplot for this data computationally expensive?

x = new_df1.iloc[:,0:7].values #independent variables
y = new_df1.iloc[:,7:].values  #dependent variable





#Split training and testing data set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)


# #Feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.fit_transform(x_test)

#sc_y = StandardScaler()
#y_train = sc_y.fit_tansform(y_train)


### Here preprocessing finish ####



####    Modeling   #####



#Fitting Simple Linear regression to the training set 
from sklearn import metrics
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV
# regressor = DecisionTreeRegressor(random_state=0)
# regressor.fit(x_train, y_train)

# #Feature scaling
# from sklearn.preprocessing import StandardScaler
# sc_x = StandardScaler()
# x_train = sc_x.fit_transform(x_train)
# x_test = sc_x.fit_transform(x_test)

#sc_y = StandardScaler()
#y_train = sc_y.fit_tansform(y_train)

# y_pred1 = regressor.predict(x_test)
# R1 = r2_score(y_test, y_pred1)
# mse1 = metrics.mean_squared_error(y_test, y_pred1)
# rmse1 = mse1**0.5
# mae1 = mean_absolute_error(y_test, y_pred1)

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

# titles = ['Permeability', 'Porosity', 'Water Saturation']
# labels = ['Measured Perm', 'Predicted Perm', 'Measured Porosity', 'Predicted Porosity', 'Measured Water Saturation', 'Predicted Water Saturation']

# for i in range(min(y_test.shape[1], y_pred4.shape[1])):
#     plt.figure(figsize=(8, 8))
    
#     actual_data = y_test[:, i]
#     predicted_data = y_pred4[:, i]

#     # Get the range of the actual data for both axes
#     data_range = max(y_test[:, i].max(), y_pred4[:, i].max())
    
#     plt.xlim(0, data_range)
#     plt.ylim(0, data_range)
    
#     # Plot the scatter plot
#     plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
#     # Plot the ideal line dynamically based on the data range
#     plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
#     # Customize plot titles and axis labels
#     plt.title(f'Scatter Plot - {titles[i]}')
#     plt.xlabel(f'{labels[2*i]}')
#     plt.ylabel(f'{labels[2*i+1]}')
    
#     plt.legend()
#     plt.show()
    
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

# test_y4 = Dtr.predict(x)
# for i in range(min(y.shape[1], test_y4.shape[1])):
#     plt.figure(figsize=(8, 8))
    
#     actual_data = y[:, i]
#     predicted_data = test_y4[:, i]

#     # Get the range of the actual data for both axes
#     data_range = max(y[:, i].max(), test_y4[:, i].max())
    
#     plt.xlim(0, data_range)
#     plt.ylim(0, data_range)
    
#     # Plot the scatter plot
#     plt.scatter(actual_data, predicted_data, label=f'{labels[2*i]} - {labels[2*i+1]}')
    
#     # Plot the ideal line dynamically based on the data range
#     plt.plot([0, data_range], [0, data_range], 'black', linestyle='--', label='Ideal Line')
    
#     # Customize plot titles and axis labels
#     plt.title(f'Scatter Plot - {titles[i]}')
#     plt.xlabel(f'{labels[2*i]}')
#     plt.ylabel(f'{labels[2*i+1]}')
    
#     plt.legend()
#     plt.show()


#Plot the whole measured data against predicted data
plt.figure(figsize = (15, 5))
plt.plot(df1['DEPT'], y[:,0], label='actual DTCO')
plt.plot(df1['DEPT'], test_Dtr_all[:,0], label='Predicted DTCO')

# plt.yscale('log')

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

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,2], label='actual Sw')
plt.plot(df2['DEPT'], test_Dtr_all[:,2], label='Predicted Sw')
plt.ylim(0, 100)
plt.title("Measured water saturation VS Predicted water saturation Using DecisionTree Regression")
plt.xlabel('Depth')
plt.ylabel('water saturation')
plt.legend()
plt.grid()    


plt.scatter(y_test, y_pred1)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

test_y1 = regressor.predict(x)
plt.scatter(y, test_y1)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,0], label='Permeability')
plt.plot(df2['DEPT'], test_y1[:,0], label='predicted permeability')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS XGBoost Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,1], label='Porosity')
plt.plot(df2['DEPT'], test_y1[:,1], label='predicted Porosity')
plt.ylim(0, 17.5)
plt.title("Measured porosity VS XGBoost Predicted porosity")
plt.xlabel('Depth')
plt.ylabel('porosity')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,2], label='water Saturation')
plt.plot(df2['DEPT'], test_y1[:,2], label='predicted water Saturation')
plt.ylim(0, 100)
plt.title("Measured permeability VS XGBoost Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('water saturation')
plt.legend()
plt.grid()
##XGBOOST ALGORITHM
pip install xgboost
import xgboost as xgb
from sklearn.model_selection import cross_val_score, KFold
xbgr = xgb.XGBRegressor()
xbgr.fit(x_train, y_train)

score = xbgr.score(x_train, y_train)  
print("Training score: ", score)

scores = cross_val_score(estimator = xbgr, X=x_train, y=y_train,cv=10)
print('Mean cross-validation score: %.2f' % scores.mean())

kfold = KFold(n_splits=10, shuffle=True)
kf_cv_scores = cross_val_score(xbgr, x_train, y_train, cv=kfold )
print("K-fold CV average score: %.2f" % kf_cv_scores.mean())

y_pred7 = xbgr.predict(x_test)
R7 = r2_score(y_test, y_pred7)
mse7 = metrics.mean_squared_error(y_test, y_pred7)
rmse7 = mse7**0.5
mae7 = mean_absolute_error(y_test, y_pred7)

plt.scatter(y_test, y_pred7)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

test_y2 = xbgr.predict(x)
plt.scatter(y, test_y2)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,0], label='Permeability')
plt.plot(df2['DEPT'], test_y2[:,0], label='predicted permeability')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS XGBoost Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,1], label='Porosity')
plt.plot(df2['DEPT'], test_y2[:,1], label='predicted Porosity')
plt.ylim(0, 17.5)
plt.title("Measured porosity VS XGBoost Predicted porosity")
plt.xlabel('Depth')
plt.ylabel('porosity')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,2], label='water Saturation')
plt.plot(df2['DEPT'], test_y2[:,2], label='predicted water Saturation')
plt.ylim(0, 100)
plt.title("Measured permeability VS XGBoost Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('water saturation')
plt.legend()
plt.grid()


pip install lightgbm
import lightgbm as lgb



train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)  

params = {
    'objective': 'regression',
    'metric': 'mse',  # Mean Squared Error
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

num_round = 1000
bst = lgb.train(params, train_set=train_data, valid_sets=[test_data], num_boost_round=num_round)


## Make Predictions
y_pred8 = bst.predict(x_test, num_iteration=bst.best_iteration)

import lightgbm as lgb

# Assuming you have x_train, y_train, x_test, y_test

# Convert the NumPy arrays to lists
y_train_list = y_train.ravel().tolist()  # Ensure y_train is 1D
y_test_list = y_test.ravel().tolist()    # Ensure y_test is 1D

# Create LightGBM datasets
train_data = lgb.Dataset(x_train, label=y_train_list)
test_data = lgb.Dataset(x_test, label=y_test_list, reference=train_data)

# Define parameters
params = {
    'objective': 'regression',
    'metric': 'mse',  # Mean Squared Error
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}

# Train the model with early stopping
num_round = 1000
bst = lgb.train(
    params,
    train_set=train_data,
    valid_sets=[test_data])

# Make Predictions
y_pred8 = bst.predict(x_test, num_iteration=bst.best_iteration)

# Evaluation of Model
R8 = r2_score(y_test_list, y_pred8)
mse8 = metrics.mean_squared_error(y_test, y_pred8)
rmse8 = mse8**0.5
mae8 = mean_absolute_error(y_test, y_pred8)

plt.scatter(y_test, y_pred8)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

test_y3 = bst.predict(x)
plt.scatter(y, test_y3)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y, label='actual Perm')
plt.plot(df2['DEPT'], test_y3, label='Predicted Perm')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS LightGBM Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()


# MLPRegressor 
from sklearn.neural_network import MLPRegressor
# #Feature scaling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)

mlp_regressor = MLPRegressor()
mlp_regressor.fit(x_train, y_train.ravel())

y_pred9 = mlp_regressor.predict(x_test)

# Evaluation of Model
R9 = r2_score(y_test, y_pred9)
mse9 = metrics.mean_squared_error(y_test, y_pred9)
rmse9 = mse9**0.5
mae9 = mean_absolute_error(y_test, y_pred9)

plt.scatter(y_test, y_pred9)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

test_y4 = mlp_regressor.predict(x)
plt.scatter(y, test_y4)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y, label='actual Perm')
plt.plot(df2['DEPT'], test_y4, label='Predicted Perm')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()
# MLP regressor is very poor. Not good for my work

#ADABOOST REGRESSOR WITH DECISION TREE REGRESSOR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Assume X, y are your feature matrix and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train_scaled = sc_x.fit_transform(x_train)
x_test_scaled = sc_x.fit_transform(x_test)

y_test_scaled = sc_x.fit_transform(x_test)
sc_y = StandardScaler()
y_train_scaled = sc_y.fit_transform(y_train.reshape(-1, 1)).ravel()

# Create a weak learner (base estimator), in this case, a Decision Tree
base_regressor = DecisionTreeRegressor(max_depth=1)

# Create AdaBoost regressor
adaboost_regressor = AdaBoostRegressor(base_regressor, n_estimators=50, random_state=42)

# Train the model
adaboost_regressor.fit(x_train, y_train)

print(x_train.shape, y_train.shape)

adaboost_regressors = []
for i in range(y_train.shape[1]):
    base_regressor = DecisionTreeRegressor(max_depth=1)
    adaboost_regressor = AdaBoostRegressor(base_regressor, n_estimators=50, random_state=42)
    adaboost_regressor.fit(x_train_scaled, y_train[:, i])
    adaboost_regressors.append(adaboost_regressor)

# Make predictions on the test set
# y_pred10 = adaboost_regressor.predict(x_test)
y_pred10 = np.column_stack([regressor.predict(x_test_scaled) for regressor in adaboost_regressors])
# Evaluate the model
mse10 = metrics.mean_squared_error(y_test, y_pred10)
print(f'Mean Squared Error: {mse10}')
R10 = r2_score(y_test, y_pred10)
print(f'RSquared: {R10}')
rmse10 = mse10**0.5
mae10 = mean_absolute_error(y_test, y_pred10)

# Iterate over each column and create scatter plots 
## LOOP FOR SCATTER PLOTS     
titles = ['Permeability', 'Porosity', 'Water Saturation']
labels = ['Measured Perm', 'Predicted Perm', 'Measured Porosity', 'Predicted Porosity', 'Measured Water Saturation', 'Predicted Water Saturation']

for i in range(min(y_test.shape[1], y_pred11.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y_test[:, i]
    predicted_data = y_pred11[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y_test[:, i].max(), y_pred11[:, i].max())
    
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
    
    
plt.scatter(y_test, y_pred10)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

test_y5 = adaboost_regressor.predict(x)
plt.scatter(y, test_y5)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')


 ##SCATTER PLOT OF ACTUAL DATA AGAINST PREDICTED DATA FOR THE ENTIRE DATA THAT IS THE PREDICTION WAS DONE ON THE ENTIRE Y
test_y5 = np.column_stack([adaboost_regressor.predict(x) for regressor in adaboost_regressors])

for i in range(min(y.shape[1], test_y5.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y[:, i]
    predicted_data = test_y5[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y[:, i].max(), test_y5[:, i].max())
    
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
    
    
    
plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y, label='actual Perm')
plt.plot(df2['DEPT'], test_y5, label='Predicted Perm')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()

#Plot the whole measured data against predicted data
plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,0], label='actual Perm')
plt.plot(df2['DEPT'], test_y5[:,0], label='Predicted Perm')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,1], label='actual Poro')
plt.plot(df2['DEPT'], test_y5[:,1], label='Predicted Poro')
plt.ylim(0, 17.5)
plt.title("Measured porosity VS Predicted porosity")
plt.xlabel('Depth')
plt.ylabel('porosity')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,2], label='actual Sw')
plt.plot(df2['DEPT'], test_y5[:,2], label='Predicted Sw')
plt.ylim(0, 100)
plt.title("Measured water saturation VS Predicted water saturation")
plt.xlabel('Depth')
plt.ylabel('water saturation')
plt.legend()
plt.grid()

# Assuming you have a DataFrame df2 containing 'DEPT', and arrays y and test_y1 with shape (n_samples, 3)
fig, axs = plt.subplots(3, 1, figsize=(15, 15))

quantities = ['Permeability', 'Porosity', 'Water Saturation']
y_labels = ['permeability', 'porosity', 'water saturation']

for i in range(3):
    axs[i].plot(df2['DEPT'], y[:, i], label=f'Measured {quantities[i]}')
    axs[i].plot(df2['DEPT'], test_y5[:, i], label=f'Predicted {quantities[i]}')
    axs[i].set_ylim(0, 100 if quantities[i] == 'Water Saturation' else 17.5 if quantities[i] == 'Porosity' else 0.9)
    axs[i].set_title(f"Measured vs Predicted {quantities[i]}")
    axs[i].set_xlabel('Depth')
    axs[i].set_ylabel(y_labels[i])
    axs[i].legend()
    axs[i].grid()
    
#ADABOOST REGRESSOR WITH Support Vector regression
from sklearn import metrics
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR
# Create a weak learner (base estimator), in this case, a Decision Tree
# base_regressor = SVR(kernel='rbf')

# # Create AdaBoost regressor
# adaboost_regressor = AdaBoostRegressor(base_regressor, n_estimators=50, random_state=42)

# # Train the model
# adaboost_regressor.fit(x_train, y_train)
# print(x_train.shape, y_train.shape)

adaboost_regressors = []
for i in range(y_train.shape[1]):
    base_regressor = SVR(kernel='rbf')
    adaboost_regressor = AdaBoostRegressor(base_regressor, n_estimators=50, random_state=42)
    adaboost_regressor.fit(x_train_scaled, y_train[:, i])
    adaboost_regressors.append(adaboost_regressor)
# Make predictions on the test set
#y_pred11 = adaboost_regressor.predict(x_test)

y_pred11 = np.column_stack([regressor.predict(x_test_scaled) for regressor in adaboost_regressors])

# Evaluate the model
mse11 = metrics.mean_squared_error(y_test, y_pred11)
print(f'Mean Squared Error: {mse11}')
R11 = r2_score(y_test, y_pred11)
print(f'RSquared: {R11}')
rmse11 = mse11**0.5
mae11 = mean_absolute_error(y_test, y_pred11)

# plt.scatter(y_test, y_pred11)
# plt.xlim(0, 0.9)   # to plot a line
# plt.ylim(0, 0.9) 
# plt.plot([0, 0.9], [0, 0.9], 'black')

# Iterate over each column and create scatter plots 
## LOOP FOR SCATTER PLOTS     
titles = ['Permeability', 'Porosity', 'Water Saturation']
labels = ['Measured Perm', 'Predicted Perm', 'Measured Porosity', 'Predicted Porosity', 'Measured Water Saturation', 'Predicted Water Saturation']

for i in range(min(y_test.shape[1], y_pred11.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y_test[:, i]
    predicted_data = y_pred11[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y_test[:, i].max(), y_pred11[:, i].max())
    
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
 
    ## OR
    
## SCATTER PLOTS
 plt.figure(figsize=(8, 8))
 plt.scatter(y_test[:,0], y_pred11[:,0], label=f'permeability')
 plt.xlim(0, 0.9)
 plt.ylim(0, 0.9)
 plt.plot([0, 0.9], [0, 0.9], 'black', label='Ideal Line')
 plt.title(f'Scatter Plot')
 plt.xlabel(f'Measured perm')
 plt.ylabel(f'Predicted perm')
 plt.legend()
 plt.show()
 plt.figure(figsize=(8, 8))
 plt.scatter(y_test[:,1], y_pred11[:,1], label=f'porosity')
 plt.xlim(0, 17.5)
 plt.ylim(0, 17.5)
 plt.plot([0, 17.5], [0, 17.5], 'black', label='Ideal Line')
 plt.title(f'Scatter Plot')
 plt.xlabel(f'Measured porosity')
 plt.ylabel(f'Predicted porosity')
 plt.legend()
 plt.show()
 plt.figure(figsize=(8, 8))
 plt.scatter(y_test[:,1], y_pred11[:,1], label=f'Water Saturation percent')
 plt.xlim(0, 100)
 plt.ylim(0, 100)
 plt.plot([0, 100], [0, 100], 'black', label='Ideal Line')
 plt.title(f'Scatter Plot')
 plt.xlabel(f'Measured water saturation')
 plt.ylabel(f'Predicted water saturation')
 plt.legend()
 plt.show()
 
 # print(y_test[:,1])
 
 # plt.axis('equal')
 # plt.gca().set_aspect('equal', adjustable='box')
 # plt.title(f'Scatter Plot for Column {i+1}')
 # plt.xlabel(f'y_test - Column {i+1}')
 # plt.ylabel(f'y_pred11 - Column {i+1}')
 # plt.legend()
 # plt.show()
 
 ##SCATTER PLOT OF ACTUAL DATA AGAINST PREDICTED DATA FOR THE ENTIRE DATA THAT IS THE PREDICTION WAS DONE ON THE ENTIRE Y
test_y6 = np.column_stack([adaboost_regressor.predict(x) for regressor in adaboost_regressors])

for i in range(min(y.shape[1], test_y6.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y[:, i]
    predicted_data = test_y6[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y[:, i].max(), test_y6[:, i].max())
    
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

# test_y5 = adaboost_regressor.predict(x)
# plt.scatter(y[:,0], test_y6[:,0])
# plt.xlim(0, 0.9)   # to plot a line
# plt.ylim(0, 0.9) 
# plt.plot([0, 0.9], [0, 0.9], 'black')

#Plot the whole measured data against predicted data
plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,0], label='actual Perm')
plt.plot(df2['DEPT'], test_y6[:,0], label='Predicted Perm')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,1], label='actual Poro')
plt.plot(df2['DEPT'], test_y6[:,1], label='Predicted Poro')
plt.ylim(0, 17.5)
plt.title("Measured porosity VS Predicted porosity")
plt.xlabel('Depth')
plt.ylabel('porosity')
plt.legend()
plt.grid()

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y[:,2], label='actual Sw')
plt.plot(df2['DEPT'], test_y6[:,2], label='Predicted Sw')
plt.ylim(0, 100)
plt.title("Measured water saturation VS Predicted water saturation")
plt.xlabel('Depth')
plt.ylabel('water saturation')
plt.legend()
plt.grid()


# Assuming you have a DataFrame df2 containing 'DEPT', and arrays y and test_y1 with shape (n_samples, 3)
fig, axs = plt.subplots(3, 1, figsize=(15, 15))

quantities = ['Permeability', 'Porosity', 'Water Saturation']
y_labels = ['permeability', 'porosity', 'water saturation']

for i in range(3):
    axs[i].plot(df2['DEPT'], y[:, i], label=f'Measured {quantities[i]}')
    axs[i].plot(df2['DEPT'], test_y6[:, i], label=f'Predicted {quantities[i]}')
    axs[i].set_ylim(0, 100 if quantities[i] == 'Water Saturation' else 17.5 if quantities[i] == 'Porosity' else 0.9)
    axs[i].set_title(f"Measured vs Predicted {quantities[i]}")
    axs[i].set_xlabel('Depth')
    axs[i].set_ylabel(y_labels[i])
    axs[i].legend()
    axs[i].grid()

plt.tight_layout()
plt.show()

from sklearn.ensemble import RandomForestRegressor
# Create a weak learner (base estimator), in this case, a Decision Tree
# base_regressor = RandomForestRegressor(max_depth=1)

# # Create AdaBoost regressor
# adaboost_regressor = AdaBoostRegressor(base_regressor, n_estimators=50, random_state=42)

# # Train the model
# adaboost_regressor.fit(x_train, y_train.ravel())

# # Make predictions on the test set
# y_pred12 = adaboost_regressor.predict(x_test)


adaboost_regressors = []
for i in range(y_train.shape[1]):
    base_regressor = RandomForestRegressor(max_depth=1)
    adaboost_regressor = AdaBoostRegressor(base_regressor, n_estimators=50, random_state=42)
    adaboost_regressor.fit(x_train, y_train[:, i])
    adaboost_regressors.append(adaboost_regressor)
    
 y_pred12 = np.column_stack([regressor.predict(x_test) for regressor in adaboost_regressors])   
# Evaluate the model
mse12 = metrics.mean_squared_error(y_test, y_pred12)
print(f'Mean Squared Error: {mse12}')
R12 = r2_score(y_test, y_pred12)
print(f'RSquared: {R12}')
rmse12 = mse12**0.5
mae12 = mean_absolute_error(y_test, y_pred12)

plt.scatter(y_test, y_pred12)
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

test_y7 = adaboost_regressor.predict(x)
plt.scatter(y[:, 0], test_y7[:, 0])
plt.xlim(0, 0.9)   # to plot a line
plt.ylim(0, 0.9) 
plt.plot([0, 0.9], [0, 0.9], 'black')

plt.scatter(y[:, 1], test_y7[:, 1])
plt.xlim(0, 17.5)   # to plot a line
plt.ylim(0, 17.5) 
plt.plot([0, 17.5], [0, 17.5], 'black')

plt.scatter(y[:, 2], test_y7[:, 2])
plt.xlim(0, 100)   # to plot a line
plt.ylim(0, 100) 
plt.plot([0, 100], [0, 100], 'black')

plt.figure(figsize = (15, 5))
plt.plot(df2['DEPT'], y, label='actual Perm')
plt.plot(df2['DEPT'], test_y7, label='Predicted Perm')
plt.ylim(0, 0.9)
plt.title("Measured permeability VS Predicted permeability")
plt.xlabel('Depth')
plt.ylabel('permeability')
plt.legend()
plt.grid()

#######
titles = ['Permeability', 'Porosity', 'Water Saturation']
labels = ['Measured Perm', 'Predicted Perm', 'Measured Porosity', 'Predicted Porosity', 'Measured Water Saturation', 'Predicted Water Saturation']

for i in range(min(y_test.shape[1], y_pred12.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y_test[:, i]
    predicted_data = y_pred12[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y_test[:, i].max(), y_pred12[:, i].max())
    
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
    
test_y7 = np.column_stack([adaboost_regressor.predict(x) for regressor in adaboost_regressors])

for i in range(min(y.shape[1], test_y7.shape[1])):
    plt.figure(figsize=(8, 8))
    
    actual_data = y[:, i]
    predicted_data = test_y7[:, i]

    # Get the range of the actual data for both axes
    data_range = max(y[:, i].max(), test_y7[:, i].max())
    
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
    plt.plot(df2['DEPT'], y[:,0], label='actual Perm')
    plt.plot(df2['DEPT'], test_y6[:,0], label='Predicted Perm')
    plt.ylim(0, 0.9)
    plt.title("Measured permeability VS Predicted permeability")
    plt.xlabel('Depth')
    plt.ylabel('permeability')
    plt.legend()
    plt.grid()

    plt.figure(figsize = (15, 5))
    plt.plot(df2['DEPT'], y[:,1], label='actual Poro')
    plt.plot(df2['DEPT'], test_y6[:,1], label='Predicted Poro')
    plt.ylim(0, 17.5)
    plt.title("Measured porosity VS Predicted porosity")
    plt.xlabel('Depth')
    plt.ylabel('porosity')
    plt.legend()
    plt.grid()

    plt.figure(figsize = (15, 5))
    plt.plot(df2['DEPT'], y[:,2], label='actual Sw')
    plt.plot(df2['DEPT'], test_y6[:,2], label='Predicted Sw')
    plt.ylim(0, 100)
    plt.title("Measured water saturation VS Predicted water saturation")
    plt.xlabel('Depth')
    plt.ylabel('water saturation')
    plt.legend()
    plt.grid()