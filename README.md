# Heart Failure Prediction & Survival Analysis

## Project Overview
This project focuses on analyzing clinical records of heart failure patients to identify key risk factors for mortality and predict survival outcomes. As a biomedical data science portfolio project, it demonstrates a complete workflow from Exploratory Data Analysis (EDA) and statistical testing to Machine Learning modeling and Survival Analysis.

The goal is to answer two main questions:
1. **What are the most significant risk factors associated with heart failure mortality?**
2. **Can we accurately predict patient survival using clinical features?**

## Dataset
The dataset used is the **Heart Failure Clinical Records Dataset**.
- **Source**: Kaggle (originally from UCI Machine Learning Repository).
- **Size**: The provided file contains 5000 records. However, EDA reveals significant duplication. After deduplication, the dataset consists of **299 unique patients**.
- **Target Variable**: `DEATH_EVENT` (0 = Survived, 1 = Deceased)
- **Key Features**:
  - `age`: Age of the patient
  - `ejection_fraction`: Percentage of blood leaving the heart at each contraction
  - `serum_creatinine`: Level of serum creatinine in the blood
  - `serum_sodium`: Level of serum sodium in the blood
  - `time`: Follow-up period (days)
  - Comorbidities: `anaemia`, `diabetes`, `high_blood_pressure`, `smoking`
 
Created by Giulio Matteucci in 2025 as biomedical data science portfolio project.

## Methodology

### 1. Exploratory Data Analysis (EDA) & Statistical Testing
- **Data Cleaning**: Identification and removal of duplicate records.
- **Distribution Analysis**: Visualized distributions of clinical variables.
- **Statistical Significance**:
  - **Mann-Whitney U Test**: Used for numeric variables to compare distributions between surviving and deceased patients.
  - **Chi-Square Test**: Used for categorical variables to assess associations with mortality.
- **Correlation Analysis**: Examined relationships between features.

### 2. Machine Learning Modeling
Two models were trained and evaluated to predict `DEATH_EVENT`:
- **Logistic Regression**: Serves as an interpretable baseline model.
- **XGBoost Classifier**: A powerful gradient boosting model to capture non-linear relationships.

**Evaluation Metrics**:
- Accuracy, Recall, F1-Score
- ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
- Stratified K-Fold Cross-Validation for robust performance estimation.

### 3. Feature Importance & Interpretability
- Analyzed **Logistic Regression Coefficients** to understand the direction and magnitude of effects.
- Analyzed **XGBoost Feature Importance** to identify the most predictive variables.

### 4. Survival Analysis
- **Custom Kaplan-Meier Implementation**: Developed a high-performance, vectorized numpy implementation of the Kaplan-Meier estimator to visualize survival probabilities over time.
- **Stratified Analysis**: Compared survival curves across different risk groups (e.g., Age groups, Ejection Fraction levels) with bootstrap confidence intervals.

## Key Findings
- **Top Risk Factors**: The analysis highlights `age`, `ejection_fraction`, `serum_creatinine`, and `time` as the most significant predictors of mortality.
- **Model Performance**: Both Logistic Regression and XGBoost provide competitive predictive performance, with XGBoost often capturing more complex interactions.
- **Survival Insights**: Patients with lower ejection fraction and higher serum creatinine levels show markedly lower survival probabilities over time.

## 💻 Project Structure
```
├── data/
│   └── heart_failure_clinical_records.csv  # Dataset
├── misc/                                   # Miscellaneous assets (images, etc.)
├── output/                                 # Generated plots and figures
├── heart_failure_challenge.ipynb           # Main analysis notebook
├── heart_failure_utils.py                  # Custom utility functions for plotting & analysis
├── environment.yml                         # Conda environment configuration
└── README.md                               # Project documentation
```

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set up the environment**:
   This project uses `conda` for dependency management.
   ```bash
   conda env create -f environment.yml
   conda activate hf_challenge
   ```

3. **Run the Notebook**:
   Launch Jupyter Notebook or VS Code to explore the analysis.
   ```bash
   jupyter notebook heart_failure_challenge.ipynb
   ```

## Dependencies
- Python 3.10+
- pandas, numpy
- matplotlib, seaborn
- scipy
- scikit-learn
- xgboost
