# AutoRegressX

AutoRegressX is an automated machine learning (AutoML) desktop application designed to simplify regression model training for datasets in CSV format. The
system identifies feature and target variables, preprocesses data (including automatic encoding of categorical features), evaluates multiple regression algorithms,
selects the best performing model using standardized metrics, and exports both the trained model and evaluation artifacts. This tool targets IoT developers and
students who need quick, interpretable regression solutions without requiring coding expertise.

## Problem Statement
Building a regression model typically requires expertise in:
- Data preprocessing
- Feature engineering
- Model selection
- Evaluation and comparison
- Serialization of trained models
  
For IoT developers and students, these steps are often barriers to applying machine learning in real use cases, especially when deploying models to cloud backends
for inference.

## Proposed Solution
AutoRegressX automates the regression workflow by allowing the user to provide a dataset in CSV format and select the target variable. The system then 
performs automatic preprocessing by detecting numeric and categorical features, encoding categorical variables using One-Hot Encoding, imputing missing 
values, and scaling features when required, such as for Support Vector Regression (SVR).

## Model Training & Evaluation
Train multiple regression models:
- Linear Regression
- Ridge Regression
- Random Forest Regression
- Support Vector Regression
- KNN Regression
  
Evaluate using:
- R² (Coefficient of Determination)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

⚠️ When the evaluation is done, the app chooses the model that has performed well among all the trained ones and it will be the one used for making the predictions.

## References
1. Feurer, Matthias, et al. “_Efficient and Robust Automated Machine Learning._” NeurIPS 2015.
2. Olson, Ryan S., et al. “_Evaluation of a Tree-based Pipeline Optimization Tool for Automating Data Science._” GECCO 2016.
3. LeDell, Evan, et al. “_H2O AutoML: Scalable Automatic Machine Learning._” AutoML Workshop, KDD 2019.
4. James, Gareth, et al. “_An Introduction to Statistical Learning._” Springer, 2013.

## Conceptual workflow diagram
<img width="1536" height="630" alt="workflow" src="https://github.com/user-attachments/assets/be34bf43-951a-4062-a812-348ad75b9f18" />
