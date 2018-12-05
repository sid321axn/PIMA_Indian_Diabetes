# Project : PIMA Indian Diabetes Prediction

## Project Overview :
In this project I have predicted the chances of diabetes using PIMA Indian dataset.This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases. The objective of the dataset is to diagnostically predict whether or not a patient has diabetes, based on certain diagnostic measurements included in the dataset. Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage.

The datasets consists of several medical predictor variables and one target variable, **Outcome**. Predictor variables includes the number of pregnancies the patient has had, their BMI, insulin level, age, and so on.

## Installations :
This project requires Python 3.x and the following Python libraries should be installed to get the project started:
- [Numpy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org/)
- [matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [xgboost](https://xgboost.readthedocs.io/en/latest/build.html)

I also reccommend to install Anaconda, a pre-packaged Python distribution that contains all of the necessary libraries and software for this project which also include jupyter notebook to run and execute [IPython Notebook](http://ipython.org/notebook.html).

## Code :
Actual code to get started with the project is provided in two files one is,```early_Diabetes_Prediction_EDA.ipynb``` for visualization part and the other one is, ```early_Diabetes_Prediction_ML_model.ipynb``` for machine learning model.

## Run :
In a terminal or command window, navigate to the top-level project directory PIMA_Indian_Diabetes/ (that contains this README) and run one of the following commands:

```ipython notebook boston_housing.ipynb```
or

```jupyter notebook boston_housing.ipynb```

This will open the Jupyter Notebook software and project file in your browser.

## About Data :

The Pima Indians Diabetes Dataset involves predicting the onset of diabetes within 5 years in Pima Indians given medical details.

It is a binary (2-class) classification problem. The number of observations for each class is not balanced. There are 768 observations with 8 input variables and 1 output variable. The variable names are as follows:

**Features**:

- **Pregnancies** - Number of times pregnant
- **Glucose** - Plasma glucose concentration a 2 hours in an oral glucose tolerance testPlasma glucose concentration a 2 hours in an oral glucose tolerance test.
- **BloodPressure**- Diastolic blood pressure (mm Hg).
- **SkinThickness**- Triceps skinfold thickness (mm).
- **Insulin**- 2-Hour serum insulin (mu U/ml).
- **BMI**- Body mass index (weight in kg/(height in m)^2).
- **DiabetesPedigreeFunction**- Diabetes pedigree function.
- **Age**- Age in years.

**Target Variable :**
- **Outcome** - Class variable 1 if patient has diagnosed diabetes and 0 if not.

## Steps to be Followed :
Following steps I have taken to apply machine learning models:

- Importing Essential Libraries.
- Data Preparation & Data Cleaning.
- Data Visualization (already done in early_Diabetes_Prediction_EDA.ipynb)
- Feature Engineering to discover essential features in the process of applying machine learning.
- Encoding Categorical Variables.
- Train Test Split
- Apply Machine Learning Algorithm
- Cross Validation
- Model Evaluation

## Model Evaluation :
I have done model evaluation based on following sklearn metric.
- [Cross Validation Score] (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_score.html)
- [Confusion Matrix] (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)
- [Plotting ROC-AUC Curve] (https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Sensitivity and Specitivity] (https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
- [Classification Error] (https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/)
