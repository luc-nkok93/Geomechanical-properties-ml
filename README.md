# Geomechanical-properties-ml
This study is to implement and validate machine learning models to predict geomechanical properties such as shear wave velocity, compressive wave velocity, Poisson's ratio and estimating Young's modulus, and shear modulus for a well located in the little knife field in the Williston basin.
# Geomechanical Property Prediction using Machine Learning

This repository contains the machine learning models and scripts developed for the manuscript **"Optimizing Geomechanical Properties Estimation Using Machine Learning: A Case Study from the Williston Basin."** The project applies machine learning algorithms to predict geomechanical properties such as compressional wave velocity (Vp), shear wave velocity (Vs), Poisson's ratio, Young's modulus, and shear modulus from well log data.

## Features
- Implementation of machine learning models: Random Forest, Decision Tree, Extreme Trees, and XGBoost.
- Dataset preprocessing and cleaning.
- Performance evaluation metrics (R², RMSE, MAE).
- Calculation of geomechanical properties using empirical equations.

## Repository Contents

- Python script for data preprocessing and cleaning are provided in the DTR_predict Vp Vs-1.py file, then implementation of the decision tree algorithm followed by evaluation. The respective scripts are name according to the algorithm used for training and evaluation.
- `example_test.py`: A sample test script demonstrating the prediction process.
- A sample dataset (`sample_data.csv`) is provided in this repository for demonstration purposes. It contains the following columns:
- Depth, Gamma Ray, Density, Porosity, AT90, NPHI, DTCO, DTSM

The full dataset used in the study is too large but follows the same structure. The full dataset is provided as '20457_HOVDEN_Data_for_Geomech' a CSV file
- `README.md`: Repository overview and usage instructions.

## Requirements
The following dependencies are required to run the scripts:
- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install the dependencies using:
```bash
pip install -r requirements.txt
