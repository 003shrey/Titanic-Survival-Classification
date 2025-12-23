# Titanic Survival Classification

A machine learning project that predicts passenger survival on the Titanic using various classification algorithms. This analysis explores the relationship between passenger characteristics and survival outcomes, implementing multiple models to identify the most effective approach.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Models and Results](#models-and-results)
- [Conclusion](#conclusion)
- [License](#license)

## Overview

This project analyzes the famous Titanic dataset to build predictive models that determine whether a passenger survived the disaster. Using historical passenger data including demographics, ticket class, and family relationships, we implement and compare multiple machine learning algorithms to achieve optimal prediction accuracy.

## Dataset

The project uses the Titanic dataset, which contains information about passengers aboard the RMS Titanic. The dataset includes the following key features:

- **survived**: Survival indicator (0 = No, 1 = Yes)
- **pclass**: Ticket class (1 = First, 2 = Second, 3 = Third)
- **sex**: Gender of the passenger
- **age**: Age in years
- **sibsp**: Number of siblings/spouses aboard
- **parch**: Number of parents/children aboard
- **fare**: Passenger fare
- **embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)
- **alone**: Whether the passenger traveled alone

The dataset originally contains 891 records with 889 records used after data cleaning.

## Features

- Comprehensive data preprocessing and cleaning
- Handling of missing values through imputation and removal
- Feature engineering and label encoding
- Multiple machine learning model implementations
- Detailed model evaluation and comparison
- Standardized feature scaling for optimal performance

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/003shrey/Titanic-Survival-Classification.git
cd Titanic-Survival-Classification
```

2. Install required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook Titanic_Survival_Model.ipynb
```

## Usage

Open the `Titanic_Survival_Model.ipynb` notebook and execute the cells sequentially to:

1. Load and explore the dataset
2. Perform data preprocessing and cleaning
3. Apply feature engineering and encoding
4. Train multiple classification models
5. Evaluate and compare model performance
6. Review the final predictions and accuracy metrics

The notebook is well-structured with clear sections for each phase of the machine learning pipeline.

## Methodology

### Data Preprocessing

1. **Missing Value Treatment**: Age column filled with mean values; rows with missing embarkation data removed
2. **Feature Selection**: Removed redundant columns (deck, alive, embark_town, class, who, adult_male)
3. **Label Encoding**: Applied to categorical variables (sex, embarked)
4. **Data Type Conversion**: Converted all features to integer type for consistency
5. **Feature Scaling**: Applied StandardScaler to normalize feature distributions

### Model Training

The dataset is split into training (80%) and testing (20%) sets. Multiple classification algorithms are trained and evaluated:

- **Logistic Regression**: Linear classification baseline
- **Random Forest Classifier**: Ensemble learning with decision trees
- **Decision Tree Classifier**: Single tree-based classification
- **Support Vector Machine (SVM)**: Kernel-based classification with RBF kernel

## Models and Results

### Performance Comparison

| Model | Accuracy | Key Observations |
|-------|----------|-----------------|
| Logistic Regression | ~80.3% | Strong baseline performance with interpretable results |
| Random Forest | ~81.5% | Improved accuracy through ensemble learning |
| Decision Tree | ~77.0% | Simpler model with moderate performance |
| **Support Vector Machine** | **~82.6%** | **Best performing model** |

### SVM Detailed Results

The Support Vector Machine with RBF kernel achieved the highest accuracy:

- **Accuracy**: 82.6%
- **Precision**: 84% (Class 0), 80% (Class 1)
- **Recall**: 88% (Class 0), 74% (Class 1)
- **F1-Score**: 86% (Class 0), 77% (Class 1)

**Confusion Matrix**:
```
[[96  13]
 [18  51]]
```

The model correctly identified 96 non-survivors and 51 survivors, with relatively low misclassification rates.

## Conclusion

This analysis demonstrates that machine learning models can effectively predict Titanic passenger survival with high accuracy. The Support Vector Machine with RBF kernel emerged as the most effective algorithm, achieving 82.6% accuracy on the test set.

Key findings from the analysis:

1. **Model Selection**: SVM outperformed other algorithms, likely due to its ability to handle non-linear relationships in the data
2. **Feature Importance**: Passenger class, gender, and fare were critical predictors of survival
3. **Data Quality**: Proper preprocessing and feature scaling significantly improved model performance
4. **Practical Application**: The model can serve as a foundation for understanding survival factors in maritime disasters

Future improvements could include hyperparameter tuning, additional feature engineering, and exploration of more advanced ensemble methods to further enhance prediction accuracy.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Shreyansh Yadav

---

**Note**: This project is developed for educational purposes to demonstrate machine learning classification techniques and data analysis workflows.
