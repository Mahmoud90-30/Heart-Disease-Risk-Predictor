# HeartGuard: Heart Disease Risk Prediction

HeartGuard is a machine learning application that predicts the likelihood of heart disease from a patient's clinical measurements. It combines a full data analysis and modeling pipeline with a desktop application that allows a user to enter patient data and receive an instant, AI-powered risk prediction.

## Overview

Cardiovascular disease is one of the leading causes of death worldwide, and early risk assessment can support timely medical intervention. This project uses the UCI Heart Failure Prediction dataset to train and compare several classification models that estimate whether a patient is likely to have heart disease based on clinical attributes such as age, blood pressure, cholesterol, and ECG results. The best-performing setup is packaged into a desktop application, HeartGuard, so the model can be used without writing any code.

## Dataset

The dataset (`heart.csv`) contains 918 patient records with 11 clinical features and 1 target variable.

| Feature | Description |
|---|---|
| Age | Age of the patient in years |
| Sex | Sex of the patient (M / F) |
| ChestPainType | Chest pain type (TA, ATA, NAP, ASY) |
| RestingBP | Resting blood pressure (mm Hg) |
| Cholesterol | Serum cholesterol (mg/dl) |
| FastingBS | Fasting blood sugar (1 if greater than 120 mg/dl, else 0) |
| RestingECG | Resting electrocardiogram results (Normal, ST, LVH) |
| MaxHR | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina (Y / N) |
| Oldpeak | ST depression induced by exercise relative to rest |
| ST_Slope | Slope of the peak exercise ST segment (Up, Flat, Down) |
| HeartDisease | Target variable (1 = heart disease, 0 = normal) |

Source: [UCI Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/amirmahdiabbootalebi/heart-disease)

## Project Workflow

The notebook (`PROJECT_ML.ipynb`) is organized into the following stages:

### 1. Data Understanding
Initial inspection of the dataset shape, structure, data types, missing values, and duplicate records.

### 2. Exploratory Data Analysis
- Distribution analysis of numerical features (Age, RestingBP, Cholesterol, MaxHR, Oldpeak) using box plots and histograms segmented by target class
- Correlation analysis using a heatmap of numerical features
- Relationship analysis between categorical features (Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope, FastingBS) and heart disease incidence
- Outlier detection using the interquartile range (IQR) method

Key insights from the analysis:
- Older patients show a higher rate of heart disease
- Exercise-induced angina is associated with a higher rate of heart disease
- Heart disease is more common among male patients in this dataset
- Asymptomatic (ASY) chest pain is associated with a higher rate of heart disease
- Flat and downward-sloping ST segments are associated with higher heart disease rates
- Fasting blood sugar above 120 mg/dl is associated with a higher rate of heart disease

### 3. Data Preprocessing
- Encoding of binary categorical variables (Sex, ExerciseAngina) using label mapping
- One-hot encoding of multi-class categorical variables (ChestPainType, RestingECG, ST_Slope)
- Correction of invalid zero values in RestingBP and Cholesterol by replacing them with the column median
- Correction of invalid negative values in Oldpeak
- Feature scaling with StandardScaler, fit on the training set only, applied before training the distance- and margin-based models (SVM, KNN) as well as Logistic Regression and Naive Bayes
- Train-test split (80 percent train, 20 percent test, with a fixed random state for reproducibility)

### 4. Model Training and Evaluation
Seven classification approaches were trained and evaluated on the same train-test split:

- Logistic Regression
- Support Vector Machine (polynomial kernel)
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Gradient Boosting
- Voting Ensemble (soft voting across Logistic Regression, SVM, Decision Tree, KNN, and Naive Bayes)

The Voting Ensemble uses soft voting, meaning it averages each base model's predicted probability rather than just its final label. This is what allows the deployed model to report a confidence score, not just a risk class, and is the model saved as `model.pkl` for the application.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.848 | 0.891 | 0.841 | 0.865 |
| SVM (polynomial kernel) | 0.870 | 0.911 | 0.860 | - |
| Decision Tree | 0.788 | - | - | - |
| K-Nearest Neighbors | 0.859 | - | - | 0.876 |
| Gradient Boosting | 0.853 | - | - | - |
| Voting Ensemble | 0.853 | 0.908 | 0.832 | 0.868 |

Note: these figures were captured from an earlier hard-voting, four-model version of the ensemble (Logistic Regression, SVM, Decision Tree, KNN), before Naive Bayes was added. Since then, the ensemble has been switched to soft voting and Naive Bayes has been added as a fifth base model, which changes what `model.pkl` actually contains. The team should re-run evaluation on the current ensemble and update this table with fresh numbers, standardized across all models (accuracy, precision, recall, F1, and confusion matrix), as noted below.

The Support Vector Machine achieved the highest standalone accuracy, while the Voting Ensemble is the model shipped in the application because it combines multiple models into one more stable, better-calibrated prediction.

## Planned Improvements

- Re-run and record evaluation metrics for the current soft-voting, five-model ensemble, and standardize accuracy, precision, recall, F1, confusion matrix, and ROC-AUC across every model
- Add cross-validation to verify result stability beyond a single train-test split
- Perform hyperparameter tuning (GridSearchCV or RandomizedSearchCV), particularly for the Decision Tree, KNN, and SVM
- Explore stacking as an alternative to voting

## Desktop Application

HeartGuard includes a desktop interface, built with PySide6 (Qt for Python), that makes the trained model accessible to non-technical users. The application collects the same clinical inputs used during training (age, sex, chest pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise angina, oldpeak, and ST slope) through a patient information form and returns a real-time prediction alongside the model's probability score.

At startup, the application loads the trained ensemble (`model.pkl`) and the fitted scaler (`scaler.pkl`). User input is encoded into the same feature columns used during training and transformed with the saved scaler before being passed to the model, so predictions run entirely offline with no server or internet connection required.

The application includes a disclaimer stating that it is an educational project and its output is not a substitute for professional medical advice.

## Technology Stack

- Python
- pandas, numpy for data handling
- matplotlib, seaborn for visualization
- scikit-learn for modeling and evaluation
- joblib for model and scaler persistence
- PySide6 (Qt for Python) for the desktop GUI

## Project Structure

```
.
├── heart.csv
├── PROJECT_ML.ipynb
├── main.py
├── model.pkl
├── scaler.pkl
└── README.md
```

## Getting Started

1. Clone the repository and install the required packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter joblib PySide6
   ```
2. To review or retrain the model, ensure `heart.csv` is in the same directory as the notebook, then open and run `PROJECT_ML.ipynb` in Jupyter Notebook or JupyterLab. Re-running the training and save cells will regenerate `model.pkl` and `scaler.pkl`.
3. To run the desktop application, launch:
   ```
   python main.py
   ```
   from the project directory. The application loads `model.pkl` and `scaler.pkl` automatically at startup.
4. Enter the patient's clinical information in the form and select "Predict Heart Disease Risk" to view the prediction and model probability.

## Disclaimer

This project is for educational purposes only. Predictions produced by this model do not constitute a medical diagnosis and should not be used as a substitute for professional medical advice, diagnosis, or treatment.

## Authors

Omar Tharwat / 
 Mahmoud Moheb /
 Seif Ahmed /
 Sondos Sobhy /
 Nada Sayed 

## License

No license has been chosen for this project yet. Until one is added, all rights are reserved by the authors, and the code should not be reused or redistributed without their permission.
