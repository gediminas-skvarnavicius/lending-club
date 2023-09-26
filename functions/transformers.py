from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
from typing import Iterable, Optional, Union
import pandas as pd
import polars as pl


class NotInImputerPolars(BaseEstimator, TransformerMixin):
    """
    Transformer for imputing values in a DataFrame by filtering out
    values not in the specified number of most frequent values
    and replacing them with the most frequent value.

    This transformer filters each specified column in the input data
    to retain only the values that are among the most frequent.
    It then replaces the remaining values with the most frequent value.

    Parameters:
    -----------
    cols : list
        List of column names to perform imputation on.
    cat_no : int
        Number of most frequent values to consider for imputation.

    Attributes:
    -----------
    filter : dict
        A dictionary containing information about the most frequent values
        for each specified column.
        The dictionary structure is as follows:
        {
            'column_name': [most_frequent_value_1, most_frequent_value_2, ...],
            ...
        }

    Methods:
    --------
    fit(X, y=None)
        Fit the imputer to the specified columns in the input data.

    transform(X)
        Transform the input data by imputing values based on the most frequent values.

    Returns:
    --------
    X : DataFrame
        A DataFrame with values imputed based on the most frequent values
        for each specified column.
    """

    def __init__(
        self,
        filter: Optional[Iterable] = None,
        cat_no: Optional[int] = None,
        fill_value: Optional[Union[int, str, float]] = None,
        most_frequent: bool = False,
    ):
        """
        Initialize the NotInImputer.

        Parameters:
            filter (Iterable, optional): Values to filter out for each column.
            If not provided, it will be computed during fitting.
            cat_no (int, optional): Number of most frequent categories to consider
            for filtering. Ignored if `filter` is provided.
            fill_value (int, str, float, optional): Value to fill in for filtered-out values.
            If not provided, it will be computed during fitting.
        """
        if filter is None and cat_no is None:
            raise ValueError("Either 'filter' or 'cat_no' must be defined.")
        self.fill_value = fill_value
        self.filter = filter
        self.cat_no = cat_no
        self.most_frequent = most_frequent

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Fit the NotInImputer to the input data.

        Parameters:
            X (pd.Series or pd.DataFrame): Input data.

        Returns:
            self
        """
        if len(X.shape) == 1:
            # Convert the Series to a DataFrame-like structure
            if hasattr(X, "name"):
                X = pl.DataFrame({X.name: X})
            else:
                X = pl.DataFrame(X)
        if not self.filter:
            self.filter = {}
            for col in X.columns:
                self.filter[col] = (
                    X[col].value_counts().sort("counts")[col][-self.cat_no :].to_list()
                )
        if self.most_frequent:
            self.fill_values = {}
            for col in X.columns:
                self.fill_value[col] = (
                    X[col].value_counts().sort("counts")[col].to_list()[-1]
                )
        else:
            self.fill_values = {}
            for col in X.columns:
                self.fill_values[col] = self.fill_value
        return self

    def transform(
        self, X: Union[pl.Series, pl.DataFrame]
    ) -> Union[pl.Series, pl.DataFrame]:
        """
        Transform the input data by filling in values.

        Parameters:
            X (pd.Series or pd.DataFrame): Input data.

        Returns:
            Union[pd.Series, pd.DataFrame]: Transformed data with filled values.
        """
        if len(X.shape) == 1:
            # Convert the Series to a DataFrame-like structure
            X = pl.DataFrame({X.name: X})
        X_filled = X
        for col in X.columns:
            X_filled = X_filled.with_columns(
                pl.col(col)
                .map_elements(
                    lambda x: self.fill_values[col] if x not in self.filter[col] else x
                )
                .alias(col)
            )
        if len(X_filled.shape) == 1:
            X_filled = X_filled.values.reshape(-1, 1)
        return X_filled
