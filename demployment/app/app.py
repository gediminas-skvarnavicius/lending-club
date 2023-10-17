from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import polars as pl
import functions.data_cleaning as dmf
import functions.feature_engineering as feats
from pydantic import BaseModel
from model import (
    model_accept_reject,
    model_joint_grade,
    model_joint_intr,
    model_joint_subgrade,
    model_single_grade,
    model_single_intr,
    model_single_subgrade,
)
from model import __version__ as model_version

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class LoanDataInitial(BaseModel):
    """
    Pydantic model for incoming prediction requests.
    """

    Application_Date: str
    Amount_Requested: float
    Loan_Title: str
    Debt_To_Income_Ratio: float
    Zip_Code: str
    State: str
    Employment_Length: str


class PredictionResponse(BaseModel):
    """
    Pydantic model for prediction responses.
    """

    Decision: str


@app.get("/")
async def home(request: Request):
    """
    Home endpoint that displays the model version.

    Args:
        request (Request): The incoming HTTP request.

    Returns:
        TemplateResponse: HTML page displaying the model version.
    """
    return templates.TemplateResponse(
        "home.html", {"request": request, "model_version": model_version}
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(data: LoanDataInitial):
    """
    Predict endpoint that makes predictions based on input data.

    Args:
        data (PredictionRequest): The input data for prediction.

    Returns:
        PredictionResponse: The prediction results.
    """
    data_df = pl.DataFrame(data.model_dump())

    data_df.columns = [
        "Application Date",
        "Amount Requested",
        "Loan Title",
        "Debt-To-Income Ratio",
        "Zip Code",
        "State",
        "Employment Length",
    ]

    data_df = dmf.clean_accepted_rejected(data_df)
    data_df = feats.date_features_accepted_rejected(data_df, "Application Date")
    data_df = feats.title_text_features(data_df)
    prediction = model_accept_reject.predict(data_df)
    decision = {0: "Rejected", 1: "Accepted"}[prediction[0]]
    return {"Decision": decision}
