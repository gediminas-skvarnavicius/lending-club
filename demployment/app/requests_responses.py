from pydantic import BaseModel, Field
from typing import Optional


class LoanDataInitial(BaseModel):
    """
    Pydantic model for incoming loan acceptance prediction requests.
    """

    Application_Date: str
    Amount_Requested: float
    Loan_Title: str
    Debt_To_Income_Ratio: float
    Zip_Code: str
    State: str
    Employment_Length: str


class LoanQualityData(BaseModel):
    """
    Pydantic model for incoming loan quality prediction requests.
    """

    id: Optional[str] = Field(..., nullable=True)
    member_id: Optional[str] = Field(..., nullable=True)
    loan_amnt: Optional[float] = Field(..., nullable=True)
    funded_amnt: Optional[float] = Field(..., nullable=True)
    funded_amnt_inv: Optional[float] = Field(..., nullable=True)
    term: Optional[str] = Field(..., nullable=True)
    installment: Optional[float] = Field(..., nullable=True)
    emp_title: Optional[str] = Field(..., nullable=True)
    emp_length: Optional[str] = Field(..., nullable=True)
    home_ownership: Optional[str] = Field(..., nullable=True)
    annual_inc: Optional[float] = Field(..., nullable=True)
    verification_status: Optional[str] = Field(..., nullable=True)
    issue_d: Optional[str] = Field(..., nullable=True)
    loan_status: Optional[str] = Field(..., nullable=True)
    pymnt_plan: Optional[str] = Field(..., nullable=True)
    url: Optional[str] = Field(..., nullable=True)
    desc: Optional[str] = Field(..., nullable=True)
    purpose: Optional[str] = Field(..., nullable=True)
    title: Optional[str] = Field(..., nullable=True)
    zip_code: Optional[str] = Field(..., nullable=True)
    addr_state: Optional[str] = Field(..., nullable=True)
    dti: Optional[float] = Field(..., nullable=True)
    delinq_2yrs: Optional[float] = Field(..., nullable=True)
    earliest_cr_line: Optional[str] = Field(..., nullable=True)
    fico_range_low: Optional[float] = Field(..., nullable=True)
    fico_range_high: Optional[float] = Field(..., nullable=True)
    inq_last_6mths: Optional[float] = Field(..., nullable=True)
    mths_since_last_delinq: Optional[float] = Field(..., nullable=True)
    mths_since_last_record: Optional[float] = Field(..., nullable=True)
    open_acc: Optional[float] = Field(..., nullable=True)
    pub_rec: Optional[float] = Field(..., nullable=True)
    revol_bal: Optional[float] = Field(..., nullable=True)
    revol_util: Optional[float] = Field(..., nullable=True)
    total_acc: Optional[float] = Field(..., nullable=True)
    initial_list_status: Optional[str] = Field(..., nullable=True)
    out_prncp: Optional[float] = Field(..., nullable=True)
    out_prncp_inv: Optional[float] = Field(..., nullable=True)
    total_pymnt: Optional[float] = Field(..., nullable=True)
    total_pymnt_inv: Optional[float] = Field(..., nullable=True)
    total_rec_prncp: Optional[float] = Field(..., nullable=True)
    total_rec_int: Optional[float] = Field(..., nullable=True)
    total_rec_late_fee: Optional[float] = Field(..., nullable=True)
    recoveries: Optional[float] = Field(..., nullable=True)
    collection_recovery_fee: Optional[float] = Field(..., nullable=True)
    last_pymnt_d: Optional[str] = Field(..., nullable=True)
    last_pymnt_amnt: Optional[float] = Field(..., nullable=True)
    next_pymnt_d: Optional[str] = Field(..., nullable=True)
    last_credit_pull_d: Optional[str] = Field(..., nullable=True)
    last_fico_range_high: Optional[float] = Field(..., nullable=True)
    last_fico_range_low: Optional[float] = Field(..., nullable=True)
    collections_12_mths_ex_med: Optional[float] = Field(..., nullable=True)
    mths_since_last_major_derog: Optional[float] = Field(..., nullable=True)
    policy_code: Optional[float] = Field(..., nullable=True)
    application_type: Optional[str] = Field(..., nullable=True)
    annual_inc_joint: Optional[float] = Field(..., nullable=True)
    dti_joint: Optional[float] = Field(..., nullable=True)
    verification_status_joint: Optional[str] = Field(..., nullable=True)
    acc_now_delinq: Optional[float] = Field(..., nullable=True)
    tot_coll_amt: Optional[float] = Field(..., nullable=True)
    tot_cur_bal: Optional[float] = Field(..., nullable=True)
    open_acc_6m: Optional[float] = Field(..., nullable=True)
    open_act_il: Optional[float] = Field(..., nullable=True)
    open_il_12m: Optional[float] = Field(..., nullable=True)
    open_il_24m: Optional[float] = Field(..., nullable=True)
    mths_since_rcnt_il: Optional[float] = Field(..., nullable=True)
    total_bal_il: Optional[float] = Field(..., nullable=True)
    il_util: Optional[float] = Field(..., nullable=True)
    open_rv_12m: Optional[float] = Field(..., nullable=True)
    open_rv_24m: Optional[float] = Field(..., nullable=True)
    max_bal_bc: Optional[float] = Field(..., nullable=True)
    all_util: Optional[float] = Field(..., nullable=True)
    total_rev_hi_lim: Optional[float] = Field(..., nullable=True)
    inq_fi: Optional[float] = Field(..., nullable=True)
    total_cu_tl: Optional[float] = Field(..., nullable=True)
    inq_last_12m: Optional[float] = Field(..., nullable=True)
    acc_open_past_24mths: Optional[float] = Field(..., nullable=True)
    avg_cur_bal: Optional[float] = Field(..., nullable=True)
    bc_open_to_buy: Optional[float] = Field(..., nullable=True)
    bc_util: Optional[float] = Field(..., nullable=True)
    chargeoff_within_12_mths: Optional[float] = Field(..., nullable=True)
    delinq_amnt: Optional[float] = Field(..., nullable=True)
    mo_sin_old_il_acct: Optional[float] = Field(..., nullable=True)
    mo_sin_old_rev_tl_op: Optional[float] = Field(..., nullable=True)
    mo_sin_rcnt_rev_tl_op: Optional[float] = Field(..., nullable=True)
    mo_sin_rcnt_tl: Optional[float] = Field(..., nullable=True)
    mort_acc: Optional[float] = Field(..., nullable=True)
    mths_since_recent_bc: Optional[float] = Field(..., nullable=True)
    mths_since_recent_bc_dlq: Optional[float] = Field(..., nullable=True)
    mths_since_recent_inq: Optional[float] = Field(..., nullable=True)
    mths_since_recent_revol_delinq: Optional[float] = Field(..., nullable=True)
    num_accts_ever_120_pd: Optional[float] = Field(..., nullable=True)
    num_actv_bc_tl: Optional[float] = Field(..., nullable=True)
    num_actv_rev_tl: Optional[float] = Field(..., nullable=True)
    num_bc_sats: Optional[float] = Field(..., nullable=True)
    num_bc_tl: Optional[float] = Field(..., nullable=True)
    num_il_tl: Optional[float] = Field(..., nullable=True)
    num_op_rev_tl: Optional[float] = Field(..., nullable=True)
    num_rev_accts: Optional[float] = Field(..., nullable=True)
    num_rev_tl_bal_gt_0: Optional[float] = Field(..., nullable=True)
    num_sats: Optional[float] = Field(..., nullable=True)
    num_tl_120dpd_2m: Optional[float] = Field(..., nullable=True)
    num_tl_30dpd: Optional[float] = Field(..., nullable=True)
    num_tl_90g_dpd_24m: Optional[float] = Field(..., nullable=True)
    num_tl_op_past_12m: Optional[float] = Field(..., nullable=True)
    pct_tl_nvr_dlq: Optional[float] = Field(..., nullable=True)
    percent_bc_gt_75: Optional[float] = Field(..., nullable=True)
    pub_rec_bankruptcies: Optional[float] = Field(..., nullable=True)
    tax_liens: Optional[float] = Field(..., nullable=True)
    tot_hi_cred_lim: Optional[float] = Field(..., nullable=True)
    total_bal_ex_mort: Optional[float] = Field(..., nullable=True)
    total_bc_limit: Optional[float] = Field(..., nullable=True)
    total_il_high_credit_limit: Optional[float] = Field(..., nullable=True)
    revol_bal_joint: Optional[str] = Field(..., nullable=True)
    sec_app_fico_range_low: Optional[str] = Field(..., nullable=True)
    sec_app_fico_range_high: Optional[str] = Field(..., nullable=True)
    sec_app_earliest_cr_line: Optional[str] = Field(..., nullable=True)
    sec_app_inq_last_6mths: Optional[str] = Field(..., nullable=True)
    sec_app_mort_acc: Optional[str] = Field(..., nullable=True)
    sec_app_open_acc: Optional[str] = Field(..., nullable=True)
    sec_app_revol_util: Optional[str] = Field(..., nullable=True)
    sec_app_open_act_il: Optional[str] = Field(..., nullable=True)
    sec_app_num_rev_accts: Optional[str] = Field(..., nullable=True)
    sec_app_chargeoff_within_12_mths: Optional[str] = Field(..., nullable=True)
    sec_app_collections_12_mths_ex_med: Optional[str] = Field(..., nullable=True)
    sec_app_mths_since_last_major_derog: Optional[str] = Field(..., nullable=True)
    hardship_flag: Optional[str] = Field(..., nullable=True)
    hardship_type: Optional[str] = Field(..., nullable=True)
    hardship_reason: Optional[str] = Field(..., nullable=True)
    hardship_status: Optional[str] = Field(..., nullable=True)
    deferral_term: Optional[str] = Field(..., nullable=True)
    hardship_amount: Optional[str] = Field(..., nullable=True)
    hardship_start_date: Optional[str] = Field(..., nullable=True)
    hardship_end_date: Optional[str] = Field(..., nullable=True)
    payment_plan_start_date: Optional[str] = Field(..., nullable=True)
    hardship_length: Optional[str] = Field(..., nullable=True)
    hardship_dpd: Optional[str] = Field(..., nullable=True)
    hardship_loan_status: Optional[str] = Field(..., nullable=True)
    orig_projected_additional_accrued_interest: Optional[str] = Field(
        ..., nullable=True
    )
    hardship_payoff_balance_amount: Optional[str] = Field(..., nullable=True)
    hardship_last_payment_amount: Optional[str] = Field(..., nullable=True)
    disbursement_method: Optional[str] = Field(..., nullable=True)
    debt_settlement_flag: Optional[str] = Field(..., nullable=True)
    debt_settlement_flag_date: Optional[str] = Field(..., nullable=True)
    settlement_status: Optional[str] = Field(..., nullable=True)
    settlement_date: Optional[str] = Field(..., nullable=True)
    settlement_amount: Optional[float] = Field(..., nullable=True)
    settlement_percentage: Optional[float] = Field(..., nullable=True)
    settlement_term: Optional[float] = Field(..., nullable=True)


class QualityResponse(BaseModel):
    """
    Pydantic model for loan quality prediction responses.
    """

    Grade: str
    SubGrade: str
    InterestRate: float


class PredictionResponse(BaseModel):
    """
    Pydantic model for loan acceptance prediction responses.
    """

    Decision: str
