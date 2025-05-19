from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
import joblib
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()
model = xgb.Booster()
model.load_model("model.xgb")
logger.info("Model loaded successfully.")

transformer = joblib.load("transformer.pkl")
logger.info("Transformer loaded successfully.")

class InputFeatures(BaseModel):
    CreditScore: float
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float
    Geography: str
    Gender: str
@app.get("/")
def read_root():
    logger.info("Root endpoint called.")
    return {"message": "Welcome to the XGBoost Prediction API!"}

@app.get("/health")
def health():
    logger.info("Health check called.")
    return {"status": "ok"}

@app.post("/predict")
def predict(data: InputFeatures):
    logger.info("Predict endpoint called.")
    logger.info(f"Input received: {data.model_dump()}")

    try:
        input_df = pd.DataFrame([{
                "CreditScore": data.CreditScore,
                "Age": data.Age,
                "Tenure": data.Tenure,
                "Balance": data.Balance,
                "NumOfProducts": data.NumOfProducts,
                "HasCrCard": data.HasCrCard,
                "IsActiveMember": data.IsActiveMember,
                "EstimatedSalary":data.EstimatedSalary,
                "Geography": data.Geography,
                "Gender": data.Gender
        }])
        logger.info(f"Input DataFrame:\n{input_df}")

        transformed = transformer.transform(input_df)

        feature_names = transformer.get_feature_names_out()

        transformed_df = pd.DataFrame(transformed, columns=feature_names)

        dmatrix = xgb.DMatrix(transformed_df)
        prediction = model.predict(dmatrix)


        logger.info(f"Prediction result: {prediction.tolist()}")
        return {"prediction": prediction.tolist()}

    except Exception as e:
        logger.exception("Prediction failed.")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 