# Task 5 - Electric Vehicle Price Prediction

## 📊 Project Overview

This project focuses on predicting the price of Electric Vehicles (EVs) using Machine Learning. The dataset contains information about different EV models, including their brand, model name, range, power, and battery capacity.

A **Ridge Regression** model is used to predict the vehicle price. The project also evaluates the model using different values of the Ridge `alpha` parameter.

## 🎯 Objectives

* Load and explore the EV car dataset.
* Understand the structure and features of the dataset.
* Check data types and missing values.
* Separate input features and the target variable.
* Process categorical and numerical features.
* Split the dataset into training and testing data.
* Apply Ridge Regression for price prediction.
* Test different values of the regularization parameter `alpha`.
* Evaluate the model using MAE, RMSE, and R².

## 🛠️ Tools & Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Ridge Regression
* Jupyter Notebook / Google Colab

## 📂 Dataset

The project uses:

```text
ev_car_India_dataset.csv
```

The dataset contains EV-related features such as:

* Brand
* Model
* Range
* Power
* Battery
* Price

The **Price** column is used as the target variable.

---

# 🔄 Project Steps

## Step 1: Import Required Libraries

The required Python libraries are imported for data handling, visualization, preprocessing, model building, and evaluation.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

Scikit-learn modules are also imported for:

* Train-test splitting
* Feature preprocessing
* Standardization
* One-hot encoding
* Ridge Regression
* Model evaluation

---

## Step 2: Load the Dataset

The EV dataset is loaded using Pandas.

```python
df = pd.read_csv("/content/ev_car_India_dataset.csv")
```

The dataset is stored in the DataFrame `df`.

---

## Step 3: View the Dataset

The first few records are displayed using:

```python
print(df.head())
```

This helps to understand the columns and values available in the dataset.

---

## Step 4: Check Dataset Shape

```python
print(df.shape)
```

This displays the number of rows and columns in the dataset.

---

## Step 5: Check Data Types

```python
print(df.dtypes)
```

This checks the data type of every column and helps identify numerical and categorical features.

---

## Step 6: Generate Statistical Summary

```python
df.describe()
```

The `describe()` function provides statistical information about the numerical columns, such as:

* Count
* Mean
* Standard deviation
* Minimum
* Maximum
* Quartiles

---

## Step 7: Check Dataset Information

```python
df.info()
```

This displays information about:

* Number of rows
* Column names
* Data types
* Non-null values
* Memory usage

---

## Step 8: Check Missing Values

```python
df.isnull().sum()
```

This checks whether any columns contain missing values.

---

## Step 9: Check Unique Models

```python
df["Model"].unique()
```

This displays the different EV models available in the dataset.

---

## Step 10: Separate Features and Target

The input features and target variable are separated.

```python
X = df.drop("Price", axis=1)
y = df["Price"]
```

Here:

* `X` contains the input features.
* `y` contains the EV price that needs to be predicted.

The target variable is:

```text
Price
```

---

## Step 11: Define Categorical and Numerical Features

The features are divided into two groups.

```python
categorical = ["Brand", "Model"]

numerical = ["Range", "Power", "Battery"]
```

### Categorical Features

* Brand
* Model

### Numerical Features

* Range
* Power
* Battery

This separation is required because categorical and numerical data need different preprocessing methods.

---

## Step 12: Preprocess the Features

A `ColumnTransformer` is used to apply different preprocessing techniques.

```python
preprocessor = ColumnTransformer([
    ("Categorical",
     OneHotEncoder(handle_unknown="ignore"),
     categorical),

    ("Numerical",
     StandardScaler(),
     numerical)
])
```

### One-Hot Encoding

`OneHotEncoder` converts categorical values such as brand and model into numerical representation.

### StandardScaler

`StandardScaler` standardizes numerical features such as Range, Power, and Battery.

This makes the numerical features suitable for the Ridge Regression model.

---

## Step 13: Split the Dataset

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The dataset is divided as:

* **80% → Training data**
* **20% → Testing data**

The training data is used to train the model, while the testing data is used to evaluate its performance.

---

## Step 14: Define Ridge Regression Parameters

Different `alpha` values are tested.

```python
alphas = [0.01, 0.1, 1, 10, 100]
results = []
```

The `alpha` value controls the regularization strength in Ridge Regression.

The project evaluates the model using these five values:

```text
0.01
0.1
1
10
100
```

---

## Step 15: Create the Machine Learning Pipeline

A pipeline is created containing preprocessing and Ridge Regression.

```python
model = Pipeline([
    ("Preprocessing", preprocessor),
    ("Ridge", Ridge(alpha=alpha))
])
```

The pipeline performs two major operations:

1. Preprocess the input data.
2. Train the Ridge Regression model.

The model is then trained using:

```python
model.fit(X_train, y_train)
```

---

## Step 16: Generate Predictions

Predictions are generated for both training and testing data.

```python
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
```

* `train_pred` → predictions for training data
* `test_pred` → predictions for testing data

---

## Step 17: Calculate Training Metrics

The model's training performance is measured using:

```python
train_mae = mean_absolute_error(y_train, train_pred)

train_rmse = np.sqrt(
    mean_squared_error(y_train, train_pred)
)

train_r2 = r2_score(y_train, train_pred)
```

The following metrics are calculated:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted prices.

### RMSE

Root Mean Squared Error measures prediction error while giving more weight to larger errors.

### R² Score

R² indicates how well the model explains the variation in the target variable.

---

## Step 18: Calculate Testing Metrics

The same evaluation is performed on the test data.

```python
test_mae = mean_absolute_error(y_test, test_pred)

test_rmse = np.sqrt(
    mean_squared_error(y_test, test_pred)
)

test_r2 = r2_score(y_test, test_pred)
```

These values help evaluate how the trained model performs on unseen data.

---

## Step 19: Store the Results

The results for each `alpha` value are stored.

```python
results.append([
    alpha,
    train_mae,
    train_rmse,
    train_r2,
    test_mae,
    test_rmse,
    test_r2
])
```

This makes it possible to compare the model performance for different regularization values.

---

## Step 20: Display the Results

```python
print(results)
```

The collected results are displayed for comparison.

---

## Step 21: Create a Results DataFrame

The results are converted into a Pandas DataFrame.

```python
results = pd.DataFrame(
    results,
    columns=[
        "Alpha",
        "Train MAE",
        "Train RMSE",
        "Train R2",
        "Test MAE",
        "Test RMSE",
        "Test R2"
    ]
)
```

This organizes the model evaluation results into a structured table.

---

## Step 22: Display Ridge Regression Results

```python
print("\nRidge Regression Results:")
print(results)
```

The final table displays the performance of Ridge Regression for the different `alpha` values.

The results include:

* Alpha
* Training MAE
* Training RMSE
* Training R²
* Testing MAE
* Testing RMSE
* Testing R²

---

# 📊 Model Evaluation

The model is evaluated using three main metrics:

| Metric | Purpose                                                        |
| ------ | -------------------------------------------------------------- |
| MAE    | Measures average prediction error                              |
| RMSE   | Measures prediction error with greater weight on larger errors |
| R²     | Measures how well the model explains the target variation      |

Both **training** and **testing** results are calculated.

---

# 🔬 Ridge Regression

Ridge Regression is a type of linear regression that includes regularization.

The regularization helps control the model complexity by adding a penalty based on the size of the model coefficients.

In this project, different values of `alpha` are tested:

```text
0.01, 0.1, 1, 10, 100
```

The resulting performance values are compared using the evaluation metrics.

---

# 📋 Project Workflow

```text
Load EV Dataset
       ↓
Explore Dataset
       ↓
Check Shape and Data Types
       ↓
Check Missing Values
       ↓
Separate Features and Target
       ↓
Identify Categorical & Numerical Columns
       ↓
One-Hot Encoding
       ↓
Standard Scaling
       ↓
Train-Test Split
       ↓
Apply Ridge Regression
       ↓
Test Different Alpha Values
       ↓
Generate Predictions
       ↓
Calculate MAE, RMSE & R²
       ↓
Compare Results
```

# 📁 Project Files

```text
Task5_ML/
│
├── Task5_ML.ipynb
├── ev_car_India_dataset.csv
└── README.md
```

# 🚀 How to Run

1. Open the notebook in **Google Colab or Jupyter Notebook**.
2. Upload `ev_car_India_dataset.csv`.
3. Run the cells in sequence.
4. Explore the dataset information and preprocessing results.
5. Train the Ridge Regression model.
6. Evaluate the model for the different `alpha` values.
7. Check the final Ridge Regression results table.

# 📌 Conclusion

This project demonstrates the use of **Ridge Regression for Electric Vehicle price prediction**. The dataset is first explored and prepared by handling categorical and numerical features separately. One-hot encoding and standard scaling are applied before training the model. Different Ridge `alpha` values are tested, and the model is evaluated using MAE, RMSE, and R² scores. The final results provide a comparison of the model's performance across the selected regularization values.
