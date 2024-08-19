# 🍷 Red Wine Quality Decision With Machine Learning

## Project Overview

This project utilizes machine learning techniques to predict the quality of red wine based on its chemical properties. By applying a RandomForestClassifier and employing techniques like SMOTE for balancing the dataset, we aim to accurately classify wines into two categories: high-quality and low-quality. The dataset used in this project is sourced from the UCI Machine Learning Repository.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)
- [Contributing](#contributing)

## Introduction

Wine quality is influenced by various factors such as acidity, sugar content, and sulfur dioxide levels. This project aims to create a model that can predict the quality of wine based on these factors using machine learning techniques.

## Dataset

The dataset contains various chemical properties of red wines, including:

- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

The dataset can be found [here]([https://archive.ics.uci.edu/ml/datasets/wine+quality](https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009)).

## Data Exploration

### 1. Dataset Overview
Basic information about the dataset was examined using describe() and info() methods.

```python
df.describe().T
df.info()
```
### 2. Missing Values Check
I checked for missing values to ensure the data's integrity.

```python
df.isnull().sum()
```
![image](https://github.com/user-attachments/assets/c65fc4b4-d533-4e82-b350-b0333f6ff658)


### 3. Correlation Matrix

I computed the correlation matrix to understand relationships between different features.

```python
import seaborn as sb
import matplotlib.pyplot as plt

corr = df.corr()
plt.figure(figsize=(8, 5))
sb.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()
```
![image](https://github.com/user-attachments/assets/5b7c13a1-fe79-4cfa-9eab-02a76847aa89)

### 4. Boxplots for Outlier Detection

Boxplots were created for each variable to detect potential outliers.

```python
plt.figure(figsize=(15, 10))
df.boxplot()
plt.title('Boxplot of All Variables')
plt.xticks(rotation=45)
plt.show()
```
![image](https://github.com/user-attachments/assets/f91877da-2495-40d8-a7c7-ec28a47baf17)

In wine, the total sulfur dioxide (SO2) level typically ranges between 10 and 200 mg/L, indicating the amount of sulfur dioxide used in the wine. However, in certain cases, such as with sweet wines that have high sugar content or wines that need to be stored for extended periods, this value can rise up to 300 mg/L. Therefore, instead of focusing on individual values, I prefer to maintain a consistent level in the wine.

On the other hand, in cases where outliers exist, such as abnormally high levels of total sulfur dioxide, citric acid, or volatile acidity, it is crucial to evaluate whether these values are a result of measurement errors or rare but plausible instances. Instead of removing all outliers, I carefully assess their potential significance, ensuring that the model remains robust while preserving important variations that could represent unique characteristics of certain wines. Additionally, I use a Random Forest Classifier for this project, which is a robust ensemble learning method. It constructs multiple decision trees and aggregates their results, making it less sensitive to outliers and noisy data. This allows the model to capture complex patterns in the wine dataset while maintaining overall stability and performance.

### 5. Scatter Plots

Scatter plots were used to visualize the relationship between wine quality and each chemical property.

```python
import seaborn as sns

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

for feature in features:
    plt.figure(figsize=(20, 6))
    sns.scatterplot(x='quality', y=feature, data=df)
    plt.title(f'{feature} vs Quality')
    plt.show()
```
![qualityGraphs](https://github.com/user-attachments/assets/d7b6c476-bcbc-4ba8-b4b1-6e106cb8b987)



## Data Preprocessing

### 1. Handling Outliers

I used the Interquartile Range (IQR) method to detect and manage outliers in the free sulfur dioxide and total sulfur dioxide features.



*Outliers in Free Sulfur Dioxide*
   ```python
   Q1_free = df['free sulfur dioxide'].quantile(0.25)
   Q3_free = df['free sulfur dioxide'].quantile(0.75)
   IQR_free = Q3_free - Q1_free
   ```
*Outliers in Total Sulfur Dioxide*
   ```python
Q1_total = df['total sulfur dioxide'].quantile(0.25)
Q3_total = df['total sulfur dioxide'].quantile(0.75)
IQR_total = Q3_total - Q1_total

outliers_total = df[(df['total sulfur dioxide'] < (Q1_total - 1.5 * IQR_total)) | (df['total sulfur dioxide'] > (Q3_total + 1.5 * IQR_total))]
   ```
**Combining Outlier Rows**
  ``` python
outlier_rows = pd.concat([outliers_free, outliers_total]).drop_duplicates()
   ```
Although the dataset contains some outliers, these values were not excessively extreme or unrealistic. Therefore, I decided to retain them in the dataset.

### 2. Feature Scaling
To ensure all features contribute equally to the model, I scaled the data using StandardScaler.
  ``` python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
   ```
### 3. Train-Test Split

The data was split into training and testing sets for model evaluation.
   ```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
   ```
### 4. Balancing the Dataset


To handle class imbalances, I applied SMOTE (Synthetic Minority Over-sampling Technique).

Initially, the model was trained without SMOTE to assess its performance under the original class distribution. Subsequently, SMOTE was used to create synthetic samples and balance the dataset. The differences in model performance between these approaches are detailed in the [Results](#results) section.

   ```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   ```
## Modeling

I trained a RandomForestClassifier model on the resampled training set.

   ```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)
   ```
## Evaluation
Model performance was evaluated using a classification report, including precision, recall, and F1-score.
   ```python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
  ```
## Results

The RandomForestClassifier demonstrated strong performance in predicting wine quality. The classification report shows that the model is both precise and robust, making it suitable for this classification task.

**Using SMOTE**

When SMOTE was applied, the model's performance improved by addressing class imbalance. The generated synthetic samples helped the model to better learn from minority classes, resulting in a more balanced and accurate prediction.

![image](https://github.com/user-attachments/assets/20424ad1-7749-4c5f-ba6f-bbea3c7ba567)

**Not using SMOTE**

Without using SMOTE, the model's performance reflected the inherent class imbalance in the dataset. The predictions were more skewed towards the majority class, which can lead to less reliable results for the minority class.

![image](https://github.com/user-attachments/assets/55510f88-9dbf-47fd-bbf1-889d4c33c668)

## Usage

To use this project, follow these steps:

1. **Clone the Repository**
```bash
git clone https://github.com/AliNadirErdil/Red-Wine-Quality-Decision-With-ML.git
```
2. **Navigate to the Project Directory**
```bash
cd Red-Wine-Quality-Decision-With-ML
```
3. **Install the required dependencies:**
```bash
pip install -r requirements.txt
```
4. **Open the Jupyter Notebook**
```bash
jupyter notebook
```
Then, open main.ipynb in JupyterLab and run the cells to execute the project.
## Contributing
Contributions are welcome! If you'd like to contribute, please fork this repository and submit a pull request.
