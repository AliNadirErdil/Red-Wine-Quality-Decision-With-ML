# Red Wine Quality Decision With ML

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

python
df.describe().T
df.info()

### 2. Missing Values Check
We checked for missing values to ensure the data's integrity.

python
print(df.isnull().sum())


### 3. Correlation Matrix

We computed the correlation matrix to understand relationships between different features.

python
import seaborn as sb
import matplotlib.pyplot as plt

corr = df.corr()
plt.figure(figsize=(8, 5))
sb.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
plt.show()

### 4. Boxplots for Outlier Detection

Boxplots were created for each variable to detect potential outliers.

python
plt.figure(figsize=(15, 10))
df.boxplot()
plt.title('Boxplot of All Variables')
plt.xticks(rotation=45)
plt.show()

### 5. Scatter Plots

Scatter plots were used to visualize the relationship between wine quality and each chemical property.

python```
import seaborn as sns

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

for feature in features:
    plt.figure(figsize=(20, 6))
    sns.scatterplot(x='quality', y=feature, data=df)
    plt.title(f'{feature} vs Quality')
    plt.show()


## Data Preprocessing

### 1. Handling Outliers

We used the Interquartile Range (IQR) method to detect and manage outliers in the free sulfur dioxide and total sulfur dioxide features.



*Outliers in Free Sulfur Dioxide*
   python
   Q1_free = df['free sulfur dioxide'].quantile(0.25)
   Q3_free = df['free sulfur dioxide'].quantile(0.75)
   IQR_free = Q3_free - Q1_free
   
*Outliers in Total Sulfur Dioxide*
   python
Q1_total = df['total sulfur dioxide'].quantile(0.25)
Q3_total = df['total sulfur dioxide'].quantile(0.75)
IQR_total = Q3_total - Q1_total

outliers_total = df[(df['total sulfur dioxide'] < (Q1_total - 1.5 * IQR_total)) | (df['total sulfur dioxide'] > (Q3_total + 1.5 * IQR_total))]
   
**Combining Outlier Rows**
   python
outlier_rows = pd.concat([outliers_free, outliers_total]).drop_duplicates()
   
### 2. Feature Scaling
To ensure all features contribute equally to the model, we scaled the data using StandardScaler.
   python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
   
### 3. Train-Test Split

The data was split into training and testing sets for model evaluation.
   python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
   
### 4. Balancing the Dataset


To handle class imbalances, we applied SMOTE (Synthetic Minority Over-sampling Technique).

   python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
   
## Modeling

We trained a RandomForestClassifier model on the resampled training set.

   python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_resampled, y_resampled)
   
## Evaluation
Model performance was evaluated using a classification report, including precision, recall, and F1-score.
   python
from sklearn.metrics import classification_report

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
  
## Results

The RandomForestClassifier demonstrated strong performance in predicting wine quality. The classification report shows that the model is both precise and robust, making it suitable for this classification task.

## Usage

To use this project, follow these steps:

1. *Clone the Repository*
   bash
   git clone https://github.com/yourusername/Red-Wine-Quality-Decision-With-ML.git
2. **Install the required dependencies:**
   bash
pip install -r requirements.txt
3. *Run the model:*
   bash
python main.py

## Contributing
Contributions are welcome! If you'd like to contribute, please fork this repository and submit a pull request.
