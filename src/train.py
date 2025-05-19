"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import mlflow.models
import mlflow.sklearn
import mlflow.xgboost
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import pickle as pkl
### Import MLflow
import mlflow
### Import logging
import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')
def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    pkl.dump(col_transf, open("src/transformer.pkl", "wb"))
    mlflow.log_artifact("src/transformer.pkl", artifact_path="transformer")
    logging.debug("Preprocessing completed and data split into train/test.")
    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    logging.debug("Model training started.")
    log_xgboost = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    log_xgboost.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    signature = mlflow.models.infer_signature(X_train, y_train)
    # Log model
    mlflow.xgboost.log_model(log_xgboost, "model", signature=signature)

    ### Log the data
    mlflow.log_artifact("dataset/Churn_Modelling.csv")
    logging.debug("Model training and logging completed.")
    return log_xgboost

def main():
    logging.info("Session started...")
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("Churn Prediction")
        mlflow.start_run(run_name="XGBoost")

        logging.info("Reading data...")
        df = pd.read_csv("dataset/Churn_Modelling.csv")

        logging.info("Preprocessing data...")
        col_transf, X_train, X_test, y_train, y_test = preprocess(df)
        logging.info("Data preprocessed successfully.")

        logging.info("Training model...")
        model = train(X_train, y_train)
        logging.info("Model trained.")

        logging.info("Evaluating model...")
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        logging.info("Evaluation metrics logged.")

        mlflow.set_tag("model", "XGBoost")

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        plt.show()

    except Exception as e:
        logging.exception(f"Pipeline failed with error: {e}")
    finally:
        logging.info("Session ended...")



if __name__ == "__main__":
    main()
