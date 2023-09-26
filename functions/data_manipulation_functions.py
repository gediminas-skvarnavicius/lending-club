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
