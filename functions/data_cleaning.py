from typing import Dict
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


# def create_bins(data: np.ndarray, number: int, log: bool = False) -> np.ndarray:
#     """Creates bins from min to max of specified data"""
#     min_val_zero = False
#     if not log:
#         bins: np.ndarray = np.linspace(data.min(), data.max(), number)
#     else:
#         min_val = data.min()
#         max_val = data.max()
#         if min_val == 0:
#             min_val = np.partition(data, 1)[1]
#             min_val_zero = True
#         bins = np.logspace(np.log10(min_val), np.log10(max_val), number)
#         if min_val_zero == True:
#             bins[0] = 0
#     return bins


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


def text_int_to_num(data: pl.DataFrame, cols):
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


def str_to_date(data: pl.DataFrame, cols, fmt):
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
        # .pipe(winsorize_column, "Debt-To-Income Ratio", 0.9)
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
        .pipe(categorize_strings_is, {"OTHER": ["NONE", "ANY"]}, "home_ownership")
        .pipe(str_to_date, ["earliest_cr_line"], "%b-%Y")
        .pipe(str_to_date, ["issue_d"], "%b-%Y")
        .pipe(replace_below_min, "dti", 0, None)
        .pipe(cast_str_to_float, "revol_bal_joint")
        .pipe(cast_str_to_float, "sec_app_revol_util")
        .pipe(drop_column, "application_type")
    )
    return df
