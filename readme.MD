# Lending Club Loan Acceptance and Quality Prediciton

## Introduciton
This Project explores the Lending Club data set available [here](https://storage.googleapis.com/335-lending-club/lending-club.zip). Its primary objectives are as follows:
1. Create models to classify loan applications into accepted and rejected categories and predict loan quality metrics for accepted applications.
2. Assess these models, make data-related decisions, and propose enhancements for data collection and further modeling.

### Task Assumptions
This project assumes that loan grades and interest rates remain constant throughout the loan term. It focuses on classifying new applications, excluding features related to ongoing loan metrics.

## Contents
The project comprises three notebooks:
* EDA.ipynb - for Exploratory Data Analysis.
* model_1_accepted_rejected.ipynb - for loan acceptance model development.
* model_2_grade.ipynb - for loan quality prediction models.

Additionally, the "functions" folder contains various auxiliary functions and classes, while the "deployment" folder contains deployment-related files. The "temp" folder stores time-consuming variables saved in joblib format.

## Methods and Results

### Exploratory Data Analysis

#### Data Overview
The dataset, with approximately 2 million and 27 million rows for rejected applications and accepted loans, respectively, was evaluated. The rejected application table featured nine features, while the accepted loan table had 151. Data cleaning was extensive due to numerous null values, free-form columns, and data type discrepancies. Temporal trends were observed, leading to data splitting into training, validation, and test sets based on time, with the earliest data for training and the newest for testing. An imbalanced target class was identified, with only around 8% of applications being accepted.
#### Loan Acceptance Model
Features' relationships with the target were analyzed graphically and statistically using chi-square and Kruskal-Wallis tests for categorical and numeric features, respectively. No features were removed during EDA.
#### Loan Quality Model
It was determined that there are two main application types: form single applicants and joint applications. It was decided that it would be more efficient to create separate models for these applicaitons.

Similarly to the fist model the features were evaluated graphically and tested using statistical tests. Only several features in the joint application model had no significant relationship with the target and were removed. Multicollinearity was assessed, with only extreme cases leading to feature removal.

### Loan Acceptance model
#### Data Preparation
Data cleaning was performed using predefined functions.
#### Feature Engineering
Two types of features were added:
Text and date features were added, including categorizing application titles by purpose and encoding the month of the year as sine and cosine transformations.
#### Imputers
Imputers retained the most popular values for categorical features, and null values were replaced with negative integers for tree-based models.
#### Encoding 
Categorical features were encoded using Ordered Labeling or Target Mean encoding.
#### Feature removal
A feature removal option with Boruta for LightGBM models was included in the pipeline, utilizing the current model with the current hyperparameters as the estimator for feature importance.
#### Models
LightGBM with different algorithms, Decision Tree and logistic regression models were tested.
#### Hyperparameter Tuning
Hyperparameters were tuned using the ray-tune library and optuna algorithm for parameter selection. Multiple iterations were performed with sample sizes for training and validation sets of 300k and 100k, with the option of early stopping if criteria were not met. The primary score used was the f1-score of the positive class.
#### Model Selection
The best-performing model, LightGBM's Random Forest implementation, was chosen based on f1-score and other classification metrics.
#### Feature Importance
The model primarily relied on employment length, requested amount, debt-to-income ratio, and purpose as decision-making features.
#### Model Performance
The model achieved an accuracy score of 96% Th f-1 score for the positive class was 0.66.
### Loan Quality Models
#### Data Preparation
Data cleaning was performed using predefined functions.
#### Feature Engineering
Some date features were moddified or additionally added:
1. The month of the year was changed into it's sine and cosine transformations.
2. Earliest credit line and last credit line was transformed to represent the day difference from the current date.
3. Current year was extracted from the date.
#### Imputers
Imputers retained the most popular values for categorical features, and null values were replaced with negative integers for tree-based models.
#### Encoding 
Categorical features were encoded using Ordered Labeling or Target Mean encoding, and boolean features were encoded using one-hot encoding. The current year was encoded as last year's mean interest rate.
#### Models
Grade prediction utilized a decision tree and a LightGBM classifier.

Subgrade prediction involved LightGBM Classifier and LightGBM Regressor with post-processing rounding.

Interest rate prediction employed a LightGBM Regressor.
#### Hyperparameter Tuning
Parameters were tuned using the ray-tune library with macro average f1-score or mean squared error as scores.
#### Results
For single applications, the accuracy was 45% with a macro average f1-score of 0.29 on new data. Subgrade prediction performed even worse at 9% and 0.06. Interest rate prediction yielded a mean squared error of 13.6. Results were slightly better for joint applications due to a lower representation of low-grade loans.
#### Feature Importance
SHAP analysis indicated the most influential features for grade and interest rate predictions, including loan term, FICO range, loan amount, annual income, debt-to-income ratio, verification status, purpose, and the number of accounts opened in the past 12 months.
## Conclusions:
*The loan acceptance model provides adequate suggestions for loan acceptance/rejection.
*The loan quality models exhibited poor performance, possibly due to factors such as low-grade loan representation, subgrade classification challenges, temporal performance variations, and the need for improved modeling techniques.
*Better/different machine learning model development approaches should be explored, and external factors should be considered for a more comprehensive analysis.
## Further Improvements:
This project offers several avenues for immediate improvement:

1. Better evaluation of model performance with calculations of losses and benefits for different types of misclassifications.
2. Probability calibration aligned with business cost-benefit analysis.
3. Enhanced oversampling techniques, ideally integrated into the pipeline and fine-tuned alongside hyperparameters.

## Deployment

The project has been deployed as a single app with two response endpoints using FastAPI. The app has been dockerized, uploaded to Google Artifact Registry, and deployed on Cloud Run. You can access it at https://loan-prediction-ubfu2xlf7q-uc.a.run.app/.

#### Local testing 
For local testing, you can use the test script located at deployment/app/test.py

#### Sample data for endpoint testing
You can use the FastAPI documentation to test the endpoints with sample data.

##### Loan Acceptance:

```yaml
{
    "Application_Date": "Sep-2017",
    "Amount_Requested": 12000.0,
    "Loan_Title": "Debt consolidation",
    "Debt_To_Income_Ratio": 27.74,
    "Zip_Code": "923xx",
    "State": "CA",
    "Employment_Length": "10+ years"
}
```
##### Loan Quality Joint Application:
```yaml
{
    "id": "140965593",
    "member_id": null,
    "loan_amnt": 16000.0,
    "funded_amnt": 16000.0,
    "funded_amnt_inv": 16000.0,
    "term": " 36 months",
    "installment": 563.62,
    "emp_title": "Head Technician",
    "emp_length": "10+ years",
    "home_ownership": "MORTGAGE",
    "annual_inc": 144000.0,
    "verification_status": "Source Verified",
    "issue_d": "Sep-2018",
    "loan_status": "Current",
    "pymnt_plan": "n",
    "url": "https://lendingclub.com/browse/loanDetail.action?loan_id=140965593",
    "desc": null,
    "purpose": "debt_consolidation",
    "title": "Debt consolidation",
    "zip_code": "967xx",
    "addr_state": "HI",
    "dti": 13.83,
    "delinq_2yrs": 2.0,
    "earliest_cr_line": "May-2006",
    "fico_range_low": 695.0,
    "fico_range_high": 699.0,
    "inq_last_6mths": 0.0,
    "mths_since_last_delinq": 13.0,
    "mths_since_last_record": null,
    "open_acc": 10.0,
    "pub_rec": 0.0,
    "revol_bal": 17389.0,
    "revol_util": 58.0,
    "total_acc": 21.0,
    "initial_list_status": "w",
    "out_prncp": 13844.31,
    "out_prncp_inv": 13844.31,
    "total_pymnt": 3381.21,
    "total_pymnt_inv": 3381.21,
    "total_rec_prncp": 2155.69,
    "total_rec_int": 1197.34,
    "total_rec_late_fee": 28.18,
    "recoveries": 0.0,
    "collection_recovery_fee": 0.0,
    "last_pymnt_d": "Mar-2019",
    "last_pymnt_amnt": 563.62,
    "next_pymnt_d": "Apr-2019",
    "last_credit_pull_d": "Mar-2019",
    "last_fico_range_high": 649.0,
    "last_fico_range_low": 645.0,
    "collections_12_mths_ex_med": 0.0,
    "mths_since_last_major_derog": 13.0,
    "policy_code": 1.0,
    "application_type": "Joint App",
    "annual_inc_joint": 213800.0,
    "dti_joint": 10.58,
    "verification_status_joint": "Source Verified",
    "acc_now_delinq": 0.0,
    "tot_coll_amt": 0.0,
    "tot_cur_bal": 701082.0,
    "open_acc_6m": 1.0,
    "open_act_il": 2.0,
    "open_il_12m": 0.0,
    "open_il_24m": 1.0,
    "mths_since_rcnt_il": 17.0,
    "total_bal_il": 16701.0,
    "il_util": 69.0,
    "open_rv_12m": 2.0,
    "open_rv_24m": 3.0,
    "max_bal_bc": 9327.0,
    "all_util": 61.0,
    "total_rev_hi_lim": 30000.0,
    "inq_fi": 1.0,
    "total_cu_tl": 1.0,
    "inq_last_12m": 0.0,
    "acc_open_past_24mths": 4.0,
    "avg_cur_bal": 70108.0,
    "bc_open_to_buy": 1164.0,
    "bc_util": 90.1,
    "chargeoff_within_12_mths": 0.0,
    "delinq_amnt": 0.0,
    "mo_sin_old_il_acct": 148.0,
    "mo_sin_old_rev_tl_op": 133.0,
    "mo_sin_rcnt_rev_tl_op": 4.0,
    "mo_sin_rcnt_tl": 4.0,
    "mort_acc": 1.0,
    "mths_since_recent_bc": 18.0,
    "mths_since_recent_bc_dlq": 13.0,
    "mths_since_recent_inq": 16.0,
    "mths_since_recent_revol_delinq": 13.0,
    "num_accts_ever_120_pd": 0.0,
    "num_actv_bc_tl": 3.0,
    "num_actv_rev_tl": 6.0,
    "num_bc_sats": 3.0,
    "num_bc_tl": 5.0,
    "num_il_tl": 11.0,
    "num_op_rev_tl": 7.0,
    "num_rev_accts": 9.0,
    "num_rev_tl_bal_gt_0": 6.0,
    "num_sats": 10.0,
    "num_tl_120dpd_2m": 0.0,
    "num_tl_30dpd": 0.0,
    "num_tl_90g_dpd_24m": 1.0,
    "num_tl_op_past_12m": 2.0,
    "pct_tl_nvr_dlq": 81.0,
    "percent_bc_gt_75": 66.7,
    "pub_rec_bankruptcies": 0.0,
    "tax_liens": 0.0,
    "tot_hi_cred_lim": 776138.0,
    "total_bal_ex_mort": 34090.0,
    "total_bc_limit": 11800.0,
    "total_il_high_credit_limit": 48998.0,
    "revol_bal_joint": "21329.0",
    "sec_app_fico_range_low": "620.0",
    "sec_app_fico_range_high": "624.0",
    "sec_app_earliest_cr_line": "May-2006",
    "sec_app_inq_last_6mths": "0.0",
    "sec_app_mort_acc": "1.0",
    "sec_app_open_acc": "10.0",
    "sec_app_revol_util": "43.8",
    "sec_app_open_act_il": "0.0",
    "sec_app_num_rev_accts": "14.0",
    "sec_app_chargeoff_within_12_mths": "0.0",
    "sec_app_collections_12_mths_ex_med": "0.0",
    "sec_app_mths_since_last_major_derog": "59.0",
    "hardship_flag": "N",
    "hardship_type": null,
    "hardship_reason": null,
    "hardship_status": null,
    "deferral_term": null,
    "hardship_amount": null,
    "hardship_start_date": null,
    "hardship_end_date": null,
    "payment_plan_start_date": null,
    "hardship_length": null,
    "hardship_dpd": null,
    "hardship_loan_status": null,
    "orig_projected_additional_accrued_interest": null,
    "hardship_payoff_balance_amount": null,
    "hardship_last_payment_amount": null,
    "disbursement_method": "Cash",
    "debt_settlement_flag": "N",
    "debt_settlement_flag_date": null,
    "settlement_status": null,
    "settlement_date": null,
    "settlement_amount": null,
    "settlement_percentage": null,
    "settlement_term": null
}
```
##### Loan Quality Single Applicant:
```yaml
{
    "id": "130964697",
    "member_id": null,
    "loan_amnt": 15000.0,
    "funded_amnt": 15000.0,
    "funded_amnt_inv": 15000.0,
    "term": " 36 months",
    "installment": 483.45,
    "emp_title": "IT Director ",
    "emp_length": "2 years",
    "home_ownership": "OWN",
    "annual_inc": 196000.0,
    "verification_status": "Source Verified",
    "issue_d": "Mar-2018",
    "loan_status": "Current",
    "pymnt_plan": "n",
    "url": "https://lendingclub.com/browse/loanDetail.action?loan_id=130964697",
    "desc": null,
    "purpose": "debt_consolidation",
    "title": "Debt consolidation",
    "zip_code": "337xx",
    "addr_state": "FL",
    "dti": 18.29,
    "delinq_2yrs": 0.0,
    "earliest_cr_line": "Jul-1998",
    "fico_range_low": 700.0,
    "fico_range_high": 704.0,
    "inq_last_6mths": 0.0,
    "mths_since_last_delinq": 65.0,
    "mths_since_last_record": null,
    "open_acc": 19.0,
    "pub_rec": 0.0,
    "revol_bal": 24243.0,
    "revol_util": 46.3,
    "total_acc": 53.0,
    "initial_list_status": "w",
    "out_prncp": 10878.5,
    "out_prncp_inv": 10878.5,
    "total_pymnt": 5301.42,
    "total_pymnt_inv": 5301.42,
    "total_rec_prncp": 4121.5,
    "total_rec_int": 1179.92,
    "total_rec_late_fee": 0.0,
    "recoveries": 0.0,
    "collection_recovery_fee": 0.0,
    "last_pymnt_d": "Feb-2019",
    "last_pymnt_amnt": 483.45,
    "next_pymnt_d": "Apr-2019",
    "last_credit_pull_d": "Mar-2019",
    "last_fico_range_high": 694.0,
    "last_fico_range_low": 690.0,
    "collections_12_mths_ex_med": 0.0,
    "mths_since_last_major_derog": null,
    "policy_code": 1.0,
    "application_type": "Individual",
    "annual_inc_joint": null,
    "dti_joint": null,
    "verification_status_joint": null,
    "acc_now_delinq": 0.0,
    "tot_coll_amt": 0.0,
    "tot_cur_bal": 534954.0,
    "open_acc_6m": 4.0,
    "open_act_il": 3.0,
    "open_il_12m": 2.0,
    "open_il_24m": 2.0,
    "mths_since_rcnt_il": 6.0,
    "total_bal_il": 113470.0,
    "il_util": 59.0,
    "open_rv_12m": 4.0,
    "open_rv_24m": 12.0,
    "max_bal_bc": 10495.0,
    "all_util": 51.0,
    "total_rev_hi_lim": 52400.0,
    "inq_fi": 4.0,
    "total_cu_tl": 1.0,
    "inq_last_12m": 7.0,
    "acc_open_past_24mths": 15.0,
    "avg_cur_bal": 31468.0,
    "bc_open_to_buy": 7368.0,
    "bc_util": 74.1,
    "chargeoff_within_12_mths": 0.0,
    "delinq_amnt": 0.0,
    "mo_sin_old_il_acct": 141.0,
    "mo_sin_old_rev_tl_op": 236.0,
    "mo_sin_rcnt_rev_tl_op": 4.0,
    "mo_sin_rcnt_tl": 4.0,
    "mort_acc": 5.0,
    "mths_since_recent_bc": 11.0,
    "mths_since_recent_bc_dlq": null,
    "mths_since_recent_inq": 6.0,
    "mths_since_recent_revol_delinq": null,
    "num_accts_ever_120_pd": 0.0,
    "num_actv_bc_tl": 4.0,
    "num_actv_rev_tl": 10.0,
    "num_bc_sats": 5.0,
    "num_bc_tl": 16.0,
    "num_il_tl": 11.0,
    "num_op_rev_tl": 14.0,
    "num_rev_accts": 37.0,
    "num_rev_tl_bal_gt_0": 10.0,
    "num_sats": 19.0,
    "num_tl_120dpd_2m": 0.0,
    "num_tl_30dpd": 0.0,
    "num_tl_90g_dpd_24m": 0.0,
    "num_tl_op_past_12m": 6.0,
    "pct_tl_nvr_dlq": 98.0,
    "percent_bc_gt_75": 75.0,
    "pub_rec_bankruptcies": 0.0,
    "tax_liens": 0.0,
    "tot_hi_cred_lim": 605228.0,
    "total_bal_ex_mort": 137713.0,
    "total_bc_limit": 28500.0,
    "total_il_high_credit_limit": 147178.0,
    "revol_bal_joint": null,
    "sec_app_fico_range_low": null,
    "sec_app_fico_range_high": null,
    "sec_app_earliest_cr_line": null,
    "sec_app_inq_last_6mths": null,
    "sec_app_mort_acc": null,
    "sec_app_open_acc": null,
    "sec_app_revol_util": null,
    "sec_app_open_act_il": null,
    "sec_app_num_rev_accts": null,
    "sec_app_chargeoff_within_12_mths": null,
    "sec_app_collections_12_mths_ex_med": null,
    "sec_app_mths_since_last_major_derog": null,
    "hardship_flag": "N",
    "hardship_type": null,
    "hardship_reason": null,
    "hardship_status": null,
    "deferral_term": null,
    "hardship_amount": null,
    "hardship_start_date": null,
    "hardship_end_date": null,
    "payment_plan_start_date": null,
    "hardship_length": null,
    "hardship_dpd": null,
    "hardship_loan_status": null,
    "orig_projected_additional_accrued_interest": null,
    "hardship_payoff_balance_amount": null,
    "hardship_last_payment_amount": null,
    "disbursement_method": "Cash",
    "debt_settlement_flag": "N",
    "debt_settlement_flag_date": null,
    "settlement_status": null,
    "settlement_date": null,
    "settlement_amount": null,
    "settlement_percentage": null,
    "settlement_term": null
}
```
