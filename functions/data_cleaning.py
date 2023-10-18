from typing import Dict, Union, List
import polars as pl

# import numpy as np


def categorize_strings_contains(
    data: pl.DataFrame, category_mappings: Dict[str, list], col_name: str
) -> pl.DataFrame:
    """
    Categorizes strings in a DataFrame column based on a mapping of categories to substrings.

    Args:
        data (pl.DataFrame): The DataFrame containing the column to categorize.
        category_mappings (Dict[str, list]): A dictionary mapping categories (strings) to lists of substrings.
        col_name (str): The name of the column to categorize.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column updated to reflect the categorized values.
    """
    for category, substrings in category_mappings.items():
        data = data.with_columns(
            pl.col(col_name)
            .map_elements(
                lambda x: category
                if any(substring in x for substring in substrings)
                else x
            )
            .alias(col_name)
        )
    return data


def categorize_strings_is(
    data: pl.DataFrame, category_mappings: Dict[str, list], col_name: str
) -> pl.DataFrame:
    """
    Categorizes strings in a DataFrame column based on a mapping of categories to exact values.

    Args:
        data (pl.DataFrame): The DataFrame containing the column to categorize.
        category_mappings (Dict[str, list]): A dictionary mapping categories (strings) to lists of substrings.
        col_name (str): The name of the column to categorize.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column updated to reflect the categorized values.
    """
    for category, values in category_mappings.items():
        data = data.with_columns(
            pl.col(col_name)
            .map_elements(lambda x: category if x in values else x)
            .alias(col_name)
        )
    return data


def lowercase_underscore_text(
    df: pl.DataFrame, col: str, new_col_name: str
) -> pl.DataFrame:
    """
    Edit a column in a Polars DataFrame by converting its values to lowercase and replacing spaces with underscores.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col (str): The name of the column to be edited.
        new_col_name (str): The name of the new column to store the edited values.

    Returns:
        pl.DataFrame: A new DataFrame with the edited column.
    """
    df = df.with_columns(pl.col(col).str.to_lowercase().alias(new_col_name))
    df = df.with_columns(pl.col(new_col_name).str.replace(" ", "_").alias(new_col_name))
    return df


def replace_below_min(
    df: pl.DataFrame, col: str, min_val: float, replacement: float
) -> pl.DataFrame:
    """
    Replace values in a specified column that are below a minimum value with a new value.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col (str): The name of the column to check and replace values in.
        min_val (float): The minimum value below which values will be replaced.
        replacement (float): The value to replace values below the minimum.

    Returns:
        pl.DataFrame: A new DataFrame with values replaced as specified.
    """
    df = df.with_columns(
        pl.when(pl.col(col) < min_val)
        .then(replacement)
        .otherwise(pl.col(col))
        .alias(col)
    )
    return df


def cut_leading_underscore(data: pl.DataFrame, cols):
    for col in cols:
        data = data.with_columns(
            pl.col(col).map_elements(lambda x: x[:-1] if x[-1] == "_" else x)
        )
    return data


def winsorize_column(
    df: pl.DataFrame, col_name: str, percentile: float
) -> pl.DataFrame:
    """
    Winsorizes a specified column in a Polars DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        col_name (str): The name of the column to be winsorized.
        percentile (float): The percentile value to use for winsorization.

    Returns:
        pd.DataFrame: A modified DataFrame with the specified column winsorized.
    """
    # Calculate the specified percentile value
    q = df[col_name].quantile(percentile)

    # Perform Winsorization using the clip_max method
    df_winsorized = df.clone()
    df_winsorized = df_winsorized.with_columns(
        pl.col(col_name).clip_max(q).alias(col_name)
    )

    return df_winsorized


def text_int_to_num(data: pl.DataFrame, cols: list) -> pl.DataFrame:
    """
    Convert text containing integers to numerical values in specified columns.

    This function takes a Polars DataFrame and a list of column names. It converts
    text values within these columns that contain integers to numerical values.

    Parameters:
    -----------
    data : pl.DataFrame
        The input Polars DataFrame.
    cols : list of str
        List of column names containing text values with integers to be converted.

    Returns:
    --------
    data : pl.DataFrame
        The input Polars DataFrame with text values containing integers converted to numerical values.
    """
    for col in cols:
        data = data.with_columns(
            pl.col(col)
            .map_elements(lambda x: "".join(filter(str.isdigit, x)))
            .cast(pl.Float32)
            .alias(col)
        )
    return data


def cast_str_to_float(data, col_name):
    return data.with_columns(pl.col(col_name).cast(pl.Float32))


def str_to_date(data: pl.DataFrame, cols: Union[str, list], fmt: str):
    """
    Convert strings to date values in specified columns using the specified format.

    This function takes a Polars DataFrame, a column name or a list of column names, and
    a format string or a list of format strings. It converts the string values within
    the specified columns to date values using the specified format(s).

    Parameters:
    -----------
    data : pl.DataFrame
        The input Polars DataFrame.
    cols : Union[str, List[str]]
        Either a column name or a list of column names containing string values to be converted.
    fmt : Union[str, List[str]]
        Either a format string or a list of format strings used for date conversion.

    Returns:
    --------
    data : pl.DataFrame
        The input Polars DataFrame with string values converted to date values in the specified columns.
    """
    for col in cols:
        if isinstance(fmt, list):
            tmp_names = []
            for i, f in enumerate(fmt):
                data = data.with_columns(
                    pl.col(col).str.to_date(f, strict=False).alias(f"date_{str(i)}")
                )
                tmp_names.append(f"date_{str(i)}")
            data = data.with_columns(pl.coalesce(tmp_names).alias(col)).drop(
                columns=tmp_names
            )
        else:
            data = data.with_columns(pl.col(col).str.to_date(fmt).alias(col))
    return data


def drop_column(df: pl.DataFrame, col_to_drop: str):
    """
    Drops a column from a polars DataFrame

    Args:
    df (pl.DataFrame): The input DataFrame.
    col (str): The name of the column to drop.

    Returns:
        pl.DataFrame: A new DataFrame with the specified column removed.
    """
    return df.drop(columns=col_to_drop)


def clean_accepted_rejected(df):
    title_categories_contains = {
        "credit_card": ["credit_card"],
        "car": ["car"],
        "debt_consolidation": ["consolid", "refinan", "debt"],
        "medical": ["medic"],
        "business": ["business"],
        "moving": ["moving", "relocation"],
        "home": ["home", "house"],
        "education": ["educ", "school", "stud", "university"],
        "green_loan": ["renew"],
    }
    title_categories_exact = {None: ["_", "", " "]}

    df = (
        df.pipe(str_to_date, ["Application Date"], ["%Y-%m-%d", "%b-%Y"])
        .pipe(lowercase_underscore_text, "Loan Title", "title")
        .pipe(categorize_strings_contains, title_categories_contains, "title")
        .pipe(categorize_strings_is, title_categories_exact, "title")
        .pipe(replace_below_min, "Amount Requested", 1, None)
        .pipe(replace_below_min, "Debt-To-Income Ratio", 0, None)
    )
    return df


def clean_accepted_single(df: pl.DataFrame):
    title_categories_contains = {
        "nurse": ["nurse"],
        "driver": ["driver"],
        "manager": ["manager"],
    }
    title_categories_is = {"nurse": ["rn"], None: ["_", "", " "]}
    df = (
        df.pipe(lowercase_underscore_text, "emp_title", "emp_title")
        .pipe(cut_leading_underscore, ["emp_title"])
        .pipe(categorize_strings_contains, title_categories_contains, "emp_title")
        .pipe(categorize_strings_is, title_categories_is, "emp_title")
        .pipe(text_int_to_num, ["term"])
        .pipe(categorize_strings_is, {"OTHER": ["NONE", "ANY"]}, "home_ownership")
        .pipe(str_to_date, ["earliest_cr_line"], "%b-%Y")
        .pipe(str_to_date, ["issue_d"], "%b-%Y")
        .pipe(str_to_date, ["last_credit_pull_d"], "%b-%Y")
        .pipe(replace_below_min, "dti", 0, None)
        .pipe(drop_column, "application_type")
    )
    return df


def clean_accepted_joint(df: pl.DataFrame):
    title_categories_contains = {
        "nurse": ["nurse"],
        "driver": ["driver"],
        "manager": ["manager"],
    }
    title_categories_is = {"nurse": ["rn"], None: ["_", "", " "]}
    df = (
        df.pipe(lowercase_underscore_text, "emp_title", "emp_title")
        .pipe(cut_leading_underscore, ["emp_title"])
        .pipe(categorize_strings_contains, title_categories_contains, "emp_title")
        .pipe(categorize_strings_is, title_categories_is, "emp_title")
        .pipe(text_int_to_num, ["term"])
        .pipe(text_int_to_num, joint_string_num_cols)
        .pipe(categorize_strings_is, {"OTHER": ["NONE", "ANY"]}, "home_ownership")
        .pipe(str_to_date, ["earliest_cr_line"], "%b-%Y")
        .pipe(str_to_date, ["issue_d"], "%b-%Y")
        .pipe(str_to_date, ["last_credit_pull_d"], "%b-%Y")
        .pipe(str_to_date, ["sec_app_earliest_cr_line"], "%b-%Y")
        .pipe(replace_below_min, "dti", 0, None)
        .pipe(cast_str_to_float, "revol_bal_joint")
        .pipe(cast_str_to_float, "sec_app_revol_util")
        .pipe(drop_column, "application_type")
    )
    return df


def remove_poor_features_joint(df: pl.DataFrame):
    df = (
        df.pipe(drop_column, poor_features)
        .pipe(drop_column, highly_correlated_features)
        .pipe(drop_column, irrelevant_features)
    )
    return df


def remove_poor_features_single(df: pl.DataFrame):
    df = (
        df.pipe(drop_column, highly_correlated_features)
        .pipe(drop_column, irrelevant_features)
        .pipe(drop_column, joint_app_features)
    )
    return df


def label_target_grades(df: pl.DataFrame):
    df = df.with_columns(pl.col("grade").map_dict(grade_mapping))
    return df


def label_target_sub_grades(df):
    df = df.with_columns(pl.col("sub_grade").map_dict(subgrade_mapping))
    return df


poor_features = [
    "disbursement_method",
    "num_tl_120dpd_2m",
    "delinq_amnt",
    "mths_since_last_major_derog",
    "mths_since_last_record",
    "pymnt_plan",
]
highly_correlated_features = [
    "funded_amnt",
    "funded_amnt_inv",
    "total_bal_ex_mort",
    "tot_hi_cred_lim",
]
irrelevant_features = [
    "id",
    "member_id",
    "desc",
    "url",
    "policy_code",
    "payment_plan_start_date",
    "hardship_flag",
    "hardship_type",
    "hardship_reason",
    "hardship_status",
    "hardship_amount",
    "hardship_start_date",
    "hardship_end_date",
    "hardship_length",
    "hardship_dpd",
    "hardship_loan_status",
    "hardship_payoff_balance_amount",
    "hardship_last_payment_amount",
    "debt_settlement_flag",
    "debt_settlement_flag_date",
    "settlement_status",
    "settlement_date",
    "settlement_amount",
    "settlement_percentage",
    "settlement_term",
    "deferral_term",
    "loan_status",
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "next_pymnt_d",
    "collections_12_mths_ex_med",
    "tot_coll_amt",
    "orig_projected_additional_accrued_interest",
    "title",
    "installment",
]

joint_app_features = [
    "annual_inc_joint",
    "dti_joint",
    "verification_status_joint",
    "revol_bal_joint",
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_earliest_cr_line",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_revol_util",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
]

joint_string_num_cols = [
    "sec_app_fico_range_low",
    "sec_app_fico_range_high",
    "sec_app_inq_last_6mths",
    "sec_app_mort_acc",
    "sec_app_open_acc",
    "sec_app_open_act_il",
    "sec_app_num_rev_accts",
    "sec_app_chargeoff_within_12_mths",
    "sec_app_collections_12_mths_ex_med",
    "sec_app_mths_since_last_major_derog",
]

grade_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}

subgrade_mapping = {
    "A1": 1,
    "A2": 2,
    "A3": 3,
    "A4": 4,
    "A5": 5,
    "B1": 6,
    "B2": 7,
    "B3": 8,
    "B4": 9,
    "B5": 10,
    "C1": 11,
    "C2": 12,
    "C3": 13,
    "C4": 14,
    "C5": 15,
    "D1": 16,
    "D2": 17,
    "D3": 18,
    "D4": 19,
    "D5": 20,
    "E1": 21,
    "E2": 22,
    "E3": 23,
    "E4": 24,
    "E5": 25,
    "F1": 26,
    "F2": 27,
    "F3": 28,
    "F4": 29,
    "F5": 30,
    "G1": 31,
    "G2": 32,
    "G3": 33,
    "G4": 34,
    "G5": 35,
}
