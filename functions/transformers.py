from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
from typing import Iterable, Optional, Union
import polars as pl
import numpy as np


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
        return_format: str = "pl",
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
        self.return_format = return_format

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
            if hasattr(X, "name"):
                X = pl.DataFrame({X.name: X})
            else:
                X = pl.DataFrame(X)
        X_filled = X
        for col in X_filled.columns:
            X_filled = X_filled.with_columns(
                pl.col(col)
                .map_elements(
                    lambda x: self.fill_values[col] if x not in self.filter[col] else x
                )
                .alias(col)
            )
        if len(X_filled.shape) == 1:
            X_filled = X_filled.values.reshape(-1, 1)
        if self.return_format == "np":
            X_filled = X_filled.to_numpy()
        return X_filled


class PolarsColumnTransformer(BaseEstimator, TransformerMixin):
    class Step:
        def __init__(self, transformer, col) -> None:
            self.transformer = transformer
            self.col = col

        def fit(self, X):
            self.transformer.fit(X)
            return self

        def transform(self, X):
            return self.transformer.transform(X)

    def __init__(self, steps: Iterable[Step]):
        self.steps = steps

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        for step in self.steps:
            step.fit(X[step.col])
        return self

    def transform(self, X: Union[pl.Series, pl.DataFrame], y=None):
        for step in self.steps:
            transformed_col = step.transform(X[step.col])
            if len(transformed_col.shape) == 1:
                if isinstance(transformed_col, np.ndarray):
                    transformed_col = pl.Series(name=step.col, values=transformed_col)
                elif isinstance(transformed_col, pl.DataFrame):
                    transformed_col = transformed_col[step.col]
                X = X.with_columns(transformed_col.alias(step.col))
            else:
                if not isinstance(transformed_col, pl.DataFrame):
                    transformed_col = pl.DataFrame(transformed_col)

                X = pl.concat(
                    [X.drop(columns=step.col), transformed_col], how="horizontal"
                )

        if len(X.shape) == 1:
            X = X.values.reshape(-1, 1)
        return X


class PolarsOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self) -> None:
        self.categories: list
        self.cats_not_in_transform: list

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        self.categories = X.unique().to_list()
        return self

    def transform(self, X: pl.Series, y=None):
        name = X.name
        self.cats_not_in_transform = [
            i for i in self.categories if i not in X.unique().to_list()
        ]
        X = X.to_dummies()
        for col in self.cats_not_in_transform:
            X = X.with_columns(
                pl.zeros(len(X), pl.Int8, eager=True).alias((f"{name}_{col}"))
            )
        return X.select(sorted(X.columns))


class PolarsOrdinalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, order: list) -> None:
        self.order = order

    def fit(self, X, y=None):
        self.map = {}
        for i, val in enumerate(self.order):
            self.map[val] = i
        return self

    def transform(self, X: pl.Series, y=None):
        X = X.map_dict(self.map)
        return X
