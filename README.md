```markdown
# Bank Customer Churn Prediction (Research Branch)

This branch is dedicated to experimentation and research on churn prediction using various machine learning models. The focus is on comparing model performance and MLflow-based tracking for reproducibility and analysis.

## Dataset

The dataset is available from Kaggle:  
https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data

Download the file `Churn_Modelling.csv` and place it inside a `dataset/` directory at the root of the project.

## Models Applied

This research branch includes training and evaluation for the following models:

1. **Logistic Regression** (as a baseline)
2. **Support Vector Machine (SVM)** with a linear kernel
3. **XGBoost Classifier** (gradient boosting)

Each model is trained and evaluated using the same feature preprocessing and target variable. Performance metrics are logged and compared via MLflow.

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
````

Contents of `requirements.txt`:

```
mlflow==2.22.0
cloudpickle==3.1.1
numpy==2.2.5
pandas==2.2.3
scikit-learn==1.6.1
scipy==1.15.2
xgboost==3.0.1
# psutil==5.9.0 (optional)
```

## How to Run

### 1. Start MLflow UI

```bash
mlflow ui
```

Open your browser at:
[http://localhost:5000](http://localhost:5000)

### 2. Run the Training Script

```bash
python src/train.py
```

The script will:

* Load and preprocess the dataset
* Train each of the three models
* Log performance metrics and parameters to MLflow
* Save the best-performing model artifact

## Evaluation Metrics

For each model, the following metrics are computed and tracked:

* Accuracy
* Precision
* Recall
* F1 Score

## Project Structure

```
project-root/
├── data/
│   └── Churn_Modelling.csv
├── src/
│   └── train.py
├── requirements.txt
└── README.md
```
