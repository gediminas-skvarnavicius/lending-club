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
