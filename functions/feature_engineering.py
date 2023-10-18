import polars as pl
import numpy as np


def month_cyclic_features(data: pl.DataFrame, col: str) -> pl.DataFrame:
    """
    Create cyclic features for the month from a datetime column.

    This function takes a Polars DataFrame and a column containing datetime values.
    It adds two new columns, 'month_sin' and 'month_cos', which represent the month
    cyclically using sine and cosine functions. These features can help capture the
    cyclical patterns in monthly data.

    Parameters:
    -----------
    data : pl.DataFrame
        The input Polars DataFrame containing datetime values.
    col : str
        The name of the column containing datetime values.

    Returns:
    --------
    data : pl.DataFrame
        The input Polars DataFrame with added 'month_sin' and 'month_cos' features.
    """
    data = data.with_columns(
        pl.col(col)
        .dt.month()
        .map_elements(lambda x: np.sin(x / 12 * 2 * np.pi))
        .alias("month_sin")
    )
    data = data.with_columns(
        pl.col(col)
        .dt.month()
        .map_elements(lambda x: np.cos(x / 12 * 2 * np.pi))
        .alias("month_cos")
    )
    return data


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


def date_difference(
    df: pl.DataFrame, date_col1: str, date_col2: str, alias_col: str
) -> pl.DataFrame:
    """
    Calculate the difference in days between two date columns and create a new column.

    This function takes a Polars DataFrame and two columns containing date values.
    It calculates the difference in days between the dates in these columns and creates
    a new column with the specified alias to store the computed differences.

    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame.
    date_col1 : str
        The name of the first date column.
    date_col2 : str
        The name of the second date column.
    alias_col : str
        The alias for the new column that will store the date differences.

    Returns:
    --------
    df : pl.DataFrame
        The input Polars DataFrame with the new column containing date differences.
    """
    df = df.with_columns((df[date_col1] - df[date_col2]).dt.days().alias(alias_col))
    return df


def get_year(df: pl.DataFrame, date_col: str) -> pl.DataFrame:
    """
    Extract the year from a date column and create a new 'year' column.

    This function takes a Polars DataFrame and a column containing date values.
    It extracts the year from the date values and creates a new 'year' column to store
    the extracted years.

    Parameters:
    -----------
    df : pl.DataFrame
        The input Polars DataFrame.
    date_col : str
        The name of the column containing date values.

    Returns:
    --------
    df : pl.DataFrame
        The input Polars DataFrame with the new 'year' column.
    """
    df = df.with_columns(pl.col(date_col).dt.year().alias("year"))
    return df


def date_features_accepted_rejected(df: pl.DataFrame, date_col: str):
    df = df.pipe(month_cyclic_features, date_col).pipe(drop_column, date_col)
    return df


def date_features(df: pl.DataFrame, date_col: str):
    df = (
        df.pipe(month_cyclic_features, date_col)
        .pipe(date_difference, date_col, "earliest_cr_line", "earliest_cr_line")
        .pipe(date_difference, date_col, "last_credit_pull_d", "last_credit_pull_d")
        .pipe(get_year, date_col)
        .pipe(drop_column, date_col)
    )
    return df


def date_features_joint(df: pl.DataFrame, date_col: str):
    df = (
        df.pipe(month_cyclic_features, date_col)
        .pipe(date_difference, date_col, "earliest_cr_line", "earliest_cr_line")
        .pipe(date_difference, date_col, "last_credit_pull_d", "last_credit_pull_d")
        .pipe(
            date_difference,
            date_col,
            "sec_app_earliest_cr_line",
            "sec_app_earliest_cr_line",
        )
        .pipe(get_year, date_col)
        .pipe(drop_column, date_col)
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
