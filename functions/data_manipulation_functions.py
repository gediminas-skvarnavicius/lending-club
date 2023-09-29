from typing import Dict
import polars as pl
import numpy as np


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


def create_bins(data: np.ndarray, number: int, log: bool = False) -> np.ndarray:
    """Creates bins from min to max of specified data"""
    min_val_zero = False
    if not log:
        bins: np.ndarray = np.linspace(data.min(), data.max(), number)
    else:
        min_val = data.min()
        max_val = data.max()
        if min_val == 0:
            min_val = np.partition(data, 1)[1]
            min_val_zero = True
        bins = np.logspace(np.log10(min_val), np.log10(max_val), number)
        if min_val_zero == True:
            bins[0] = 0
    return bins


import polars as pl


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


def text_contains_numbers(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Check if a column in a Polars DataFrame contains numbers and create a new Boolean column indicating the result.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col (str): The name of the column to check.

    Returns:
        pl.DataFrame: A new DataFrame with a Boolean column indicating whether
        the input column contains numbers.
    """
    df = df.with_columns(df[col].str.contains(r"\d").alias(col + "_contains_numbers"))
    return df


def text_length(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Calculate the length of each value in a column and create a new
    column with the lengths.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col (str): The name of the column for which to calculate lengths.

    Returns:
        pl.DataFrame: A new DataFrame with a column containing the lengths of
        the values in the input column.
    """
    df = df.with_columns(df[col].str.n_chars().alias(col + "_length"))
    return df


def starts_with_lowercase(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Check if each value in a column starts with a lowercase letter or is empty
    (including a single whitespace).
    Create a new Boolean column indicating the result.

    Args:
        df (pl.DataFrame): The input DataFrame.
        col (str): The name of the column to check.

    Returns:
        pl.DataFrame: A new DataFrame with a Boolean column indicating whether
        each value meets the specified condition.
    """
    df = df.with_columns(
        df[col].str.contains(r"^(?:[a-z]|$|\s)").alias(col + "_starts_with_lowercase")
    )
    return df


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


def text_preprocessing_wo_training(df):
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
    title_categories_exact = {None: ["_", "other"]}

    df = (
        df.pipe(lowercase_underscore_text, "Loan Title", "title")
        .pipe(categorize_strings_contains, title_categories_contains, "title")
        .pipe(categorize_strings_is, title_categories_exact, "title")
        .pipe(replace_below_min, "Amount Requested", 1, None)
        .pipe(replace_below_min, "Debt-To-Income Ratio", 0, None)
        .pipe(winsorize_column, "Debt-To-Income Ratio", 0.9)
    )
    return df


def title_text_features(df: pl.DataFrame):
    df = (
        df.pipe(text_contains_numbers, "title")
        .pipe(text_length, "title")
        .pipe(starts_with_lowercase, "title")
        .pipe(drop_column, "Loan Title")
    )
    return df


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
