from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import polars as pl
import functions.data_cleaning as dmf
import functions.feature_engineering as feats
from requests_responses import (
    LoanDataInitial,
    PredictionResponse,
    LoanQualityData,
    QualityResponse,
)
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

inv_grade_map = {v: k for k, v in dmf.grade_mapping.items()}
inv_subgrade_map = {v: k for k, v in dmf.subgrade_mapping.items()}


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


@app.post("/loan_acceptance", response_model=PredictionResponse)
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


@app.post("/loan_quality", response_model=QualityResponse)
async def secondary_predict_endpoint(data: LoanQualityData):
    data_df = pl.DataFrame(data.model_dump())
    print(data_df["application_type"][0])
    if data_df["application_type"][0] == "Joint App":
        data_df = dmf.clean_accepted_joint(data_df)
        data_df = dmf.remove_poor_features_joint(data_df)
        data_df = feats.date_features_joint(data_df, "issue_d")
        grade = model_joint_grade.predict(data_df)[0]
        sub_grade = model_joint_subgrade.predict(data_df)[0]
        int_rate = model_joint_intr.predict(data_df)[0]

    elif data_df["application_type"][0] == "Individual":
        data_df = dmf.clean_accepted_single(data_df)
        data_df = dmf.remove_poor_features_single(data_df)
        data_df = feats.date_features(data_df, "issue_d")
        grade = model_single_grade.predict(data_df)[0]
        sub_grade = model_single_subgrade.predict(data_df)[0]
        int_rate = model_single_intr.predict(data_df)[0]

    else:
        pass

    response = {
        "Grade": inv_grade_map[grade],
        "SubGrade": inv_subgrade_map[sub_grade],
        "InterestRate": int_rate,
    }
    return response
