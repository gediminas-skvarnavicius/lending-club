from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
from typing import Iterable, Optional, Union, List
import polars as pl
import numpy as np
from collections import OrderedDict


class NotInImputerPolars(BaseEstimator, TransformerMixin):
    """
    Transformer for imputing values in a Polars DataFrame by filtering out
    values not in the specified number of most frequent values
    and replacing them with the most frequent value.

    This transformer filters each specified column in the input data
    to retain only the values that are among the most frequent.
    It then replaces the remaining values with the most frequent value.

    Parameters:
    -----------
    filter : Optional[Iterable]
        Values to filter out for each column. If not provided, it will be
        computed during fitting.
    cat_no : Optional[int]
        Number of most frequent categories to consider for filtering.
        Ignored if `filter` is provided.
    fill_value : Optional[Union[int, str, float]]
        Value to fill in for filtered-out values. If not provided, it will
        be computed during fitting.
    most_frequent : bool
        If True, replace missing values with the most frequent value in
        each column.
    return_format : str
        Output format. Can be 'pl' for Polars DataFrame or 'np'
        for NumPy array.

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
    fit(X)
        Fit the imputer to the specified columns in the input data.

    transform(X)
        Transform the input data by imputing values based on the most frequent values.

    Returns:
    --------
    X : pl.DataFrame or np.ndarray
        Transformed data with filled values in the specified format.
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
            cat_no (int, optional): Number of most frequent categories to
            consider for filtering. Ignored if `filter` is provided.
            fill_value (int, str, float, optional): Value to fill in for
            filtered-out values. If not provided,
            it will be computed during fitting.
        """
        if filter is None and cat_no is None:
            raise ValueError("Either 'filter' or 'cat_no' must be defined.")
        self.fill_value = fill_value
        self.filter = filter
        self.cat_no = cat_no
        self.most_frequent = most_frequent
        self.return_format = return_format

    def fit(self, X: Union[pl.Series, pl.DataFrame]):
        """
        Fit the NotInImputer to the input data.

        Parameters:
            X (pl.Series or pl.DataFrame): Input data.

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
            X (pl.Series or pl.DataFrame): Input data.

        Returns:
            Union[pl.Series, pl.DataFrame]: Transformed data with
            filled values.
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
            X_filled = X_filled.to_numpy()
        if self.return_format == "np":
            X_filled = X_filled.to_numpy()
        return X_filled


class PolarsColumnTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer for performing a specified sequence of transformations
    on Polars DataFrames.

    This transformer applies a series of transformations, each associated
    with a specific column, to the input Polars DataFrame.
    The transformations are specified as a list of 'Step' objects
    and can include any Polars or custom transformations.

    Parameters:
    -----------
    steps : Iterable[Step]
        List of 'Step' objects, each defining a transformation to
        apply to a specific column.
    step_params : dict, optional
        Dictionary of parameters for each step.

    Attributes:
    -----------
    steps : OrderedDict
        A dictionary containing the 'Step' objects and their
        associated transformations.
    step_params : dict
        Dictionary of parameters for each step.

    Methods:
    --------
    fit(X, y=None)
        Fit the transformer to the input data.

    transform(X)
        Transform the input Polars DataFrame using the specified
        sequence of transformations.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame.
    """

    class Step:
        def __init__(self, name, transformer, col) -> None:
            """
            Initialize a transformation step.

            Parameters:
            -----------
            name : str
                Name of the step.
            transformer
                Transformer to apply to the specified column.
            col : str
                Name of the column to apply the transformation to.

            Returns:
            --------
            None
            """
            self.transformer = transformer
            self.col = col
            self.name = name

        def fit(self, X, y=None):
            """
            Fit the transformer in the step to the input data.

            Parameters:
            -----------
            X : pl.Series or pl.DataFrame
                Input data.
            y : None
                Ignored. It is not used in the fitting process.

            Returns:
            --------
            self
            """
            self.transformer.fit(X, y)
            return self

        def transform(self, X):
            """
            Transform the input data using the transformer in the step.

            Parameters:
            -----------
            X : pl.Series or pl.DataFrame
                Input data.

            Returns:
            --------
            Transformed data.
            """
            return self.transformer.transform(X)

    def __init__(self, steps: Iterable[Step], step_params={}):
        """
        Initialize the PolarsColumnTransformer.

        Parameters:
        -----------
        steps : Iterable[Step]
            List of transformation steps to be applied.
        step_params : dict, optional
            Dictionary of parameters for each step.

        Returns:
        --------
        None
        """
        self.steps = OrderedDict()
        for step in steps:
            self.steps[step.name] = step
        self.step_params = step_params

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Fit the PolarsColumnTransformer to the input data.

        Parameters:
        -----------
        X : pl.Series or pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self
        """
        if self.step_params:
            for id, params in self.step_params.items():
                self.steps[id].transformer.set_params(**params)

        for step in self.steps.values():
            step.fit(X[step.col], y)
        return self

    def transform(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Transform the input data using the specified steps.

        Parameters:
        -----------
        X : pl.Series or pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame.
        """
        for step in self.steps.values():
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
    """
    One-hot encoder for Polars DataFrames.

    This encoder converts categorical columns into one-hot encoded columns.
    The resulting DataFrame has binary columns for each category, indicating
    the presence or absence of the category.

    Parameters:
    -----------
    drop : bool, default=False
        Whether to drop one of the binary columns to avoid multicollinearity.
        If True, one binary column for each category is dropped.

    Attributes:
    -----------
    categories : list
        List of unique categories found in the fitted data.
    cats_not_in_transform : list
        List of categories not found in the transformed data (if any).
    drop : bool
        Whether to drop one binary column for each category.

    Methods:
    --------
    fit(X, y=None)
        Fit the encoder to the input data.

    transform(X, y=None)
        Transform the input data into one-hot encoded format.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame with one-hot encoded columns.
    """

    def __init__(self, drop: bool = False) -> None:
        """
        Initialize the PolarsOneHotEncoder.

        Parameters:
        -----------
        drop : bool, default=False
            Whether to drop one of the binary columns to avoid
            multicollinearity.

        Returns:
        --------
        None
        """
        self.categories: list
        self.cats_not_in_transform: list
        self.drop = drop

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        """
        Fit the one-hot encoder to the input data.

        Parameters:
        -----------
        X : pl.Series or pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self
        """
        self.categories = X.unique().to_list()
        return self

    def transform(self, X: pl.Series, y=None):
        """
        Transform the input data into one-hot encoded format.

        Parameters:
        -----------
        X : pl.Series
            Input data to be one-hot encoded.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame with one-hot encoded columns.
        """
        name = X.name
        self.cats_not_in_transform = [
            i for i in self.categories if i not in X.unique().to_list()
        ]
        X = X.to_dummies()
        for col in self.cats_not_in_transform:
            X = X.with_columns(
                pl.zeros(len(X), pl.Int8, eager=True).alias((f"{name}_{col}"))
            )
        X = X.select(sorted(X.columns))
        if self.drop:
            X = X[:, ::2]
        return X


class PolarsOrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Ordinal encoder for Polars DataFrames.

    This encoder maps categorical values to ordinal values based
    on a predefined order. The input data is transformed to reflect
    the specified order of categories.

    Parameters:
    -----------
    order : List
        Predefined order for the categorical values.

    Attributes:
    -----------
    order : List
        Predefined order for the categorical values.
    map : dict
        A mapping of categorical values to their corresponding ordinal values.

    Methods:
    --------
    fit(X, y=None)
        Fit the encoder to the input data.

    transform(X, y=None)
        Transform the input data to reflect the predefined order.

    Returns:
    --------
    X : pl.Series
        Transformed Polars Series with ordinal values based on the predefined
        order.
    """

    def __init__(self, order: List) -> None:
        """
        Initialize the PolarsOrdinalEncoder.

        Parameters:
        -----------
        order : List
            Predefined order for the categorical values.

        Returns:
        --------
        None
        """
        self.order = order

    def fit(self, X, y=None):
        """
        Fit the ordinal encoder to the input data.

        Parameters:
        -----------
        X : pl.Series
            Input data.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self
        """
        self.map = {}
        for i, val in enumerate(self.order):
            self.map[val] = i
        return self

    def transform(self, X: pl.Series, y=None):
        """
        Transform the input data to reflect the predefined order.

        Parameters:
        -----------
        X : pl.Series
            Input data to be transformed.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.Series
            Transformed Polars Series with ordinal values based on the
            predefined order.
        """
        X = X.map_dict(self.map)
        return X


class PolarsNullImputer(BaseEstimator, TransformerMixin):
    """
    Null imputer for Polars DataFrames.

    This imputer replaces null (missing) values in the input data with
    specified fill values.

    Parameters:
    -----------
    fill_value : List
        List of fill values to replace null values in each column.

    Attributes:
    -----------
    fill_value : List
        List of fill values to be used for imputation.

    Methods:
    --------
    fit(X, y=None)
        Fit the imputer to the input data.

    transform(X, y=None)
        Transform the input data by replacing null values with the specified
        fill values.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame with null values replaced by fill values.
    """

    def __init__(self, fill_value: List) -> None:
        """
        Initialize the PolarsNullImputer.

        Parameters:
        -----------
        fill_value : List
            List of fill values to replace null values in each column.

        Returns:
        --------
        None
        """
        self.fill_value = fill_value

    def fit(self, X, y=None):
        """
        Fit the null imputer to the input data.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self
        """
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Transform the input data by replacing null values with the specified
        fill values.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data to be imputed.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame with null values replaced by
            fill values.
        """
        if not isinstance(X, pl.DataFrame):
            X = pl.DataFrame(X)
        bool_cols = X.select(pl.col(pl.Boolean)).columns
        for col in bool_cols:
            X = X.with_columns(pl.col(col).cast(pl.Int32).alias(col))
        X = X.fill_null(self.fill_value)
        return X


class TargetMeanOrderedLabeler(BaseEstimator, TransformerMixin):
    """
    Transformer for labeling categorical values based on their mean target
    values.

    This transformer labels categorical values based on their mean
    target values. The labels can be determined using different methods,
    such as 'label' (integer labels ordered by target mean), 'mean'
    (mean target values),
    or 'last_mean' (mean target values plus one for last year means).

    Parameters:
    -----------
    how : str, optional (default='mean')
        Method for determining labels. Accepted values are
        'label', 'mean', or 'last_mean'.

    Attributes:
    -----------
    map : dict
        A mapping of categorical values to their corresponding labels.
    how : str
        The method used to determine labels.

    Methods:
    --------
    fit(X, y)
        Fit the labeler to the input data.

    transform(X, y=None)
        Transform the input data by labeling categorical values.

    Returns:
    --------
    X : pl.Series
        Transformed Polars Series with labeled categorical values.
    """

    def __init__(self, how: str = "mean") -> None:
        """
        Initialize the TargetMeanOrderedLabeler.

        Parameters:
        -----------
        how : str, optional (default='mean')
            Method for determining labels. Accepted values are
            'label', 'mean', or 'last_mean'.

        Returns:
        --------
        None
        """
        self.map = {}
        self.how = how

    def fit(self, X: pl.Series, y: pl.Series):
        """
        Fit the labeler to the input data.

        Parameters:
        -----------
        X : pl.Series
            Categorical feature values.
        y : pl.Series
            Target values for the corresponding features.

        Returns:
        --------
        self : TargetMeanOrderedLabeler
            The fitted labeler instance.
        """
        self.map = {}
        self.sort_df = pl.DataFrame([X, y]).group_by(X.name).mean().sort(y.name)

        if self.how not in ["label", "mean", "last_mean"]:
            raise ValueError(
                """Invalid value for 'how' argument.
                Accepted values are 'label', 'mean', or 'last_mean'."""
            )

        if self.how == "label":
            for i, val in enumerate(self.sort_df[X.name]):
                self.map[val] = i
        if self.how == "mean":
            for mean, val in zip(self.sort_df[y.name], self.sort_df[X.name]):
                self.map[val] = mean
        if self.how == "last_mean":
            for mean, val in zip(self.sort_df[y.name], self.sort_df[X.name]):
                self.map[val + 1] = mean
                self.map[self.sort_df[X.name].min()] = None
        return self

    def transform(self, X: pl.Series, y=None):
        """
        Transform the input data by labeling categorical values.

        Parameters:
        -----------
        X : pl.Series
            Categorical feature values to be labeled.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.Series
            Transformed Polars Series with labeled categorical values.
        """
        X = X.map_dict(self.map)
        return X


class FeatureRemover(BaseEstimator, TransformerMixin):
    """
    Transformer for removing specified features from a Polars DataFrame.

    This transformer removes the specified columns (features)
    from a Polars DataFrame.

    Parameters:
    -----------
    feats_to_drop : Iterable of str, optional (default=[])
        List of column names to be removed from the DataFrame.

    Attributes:
    -----------
    feats_to_drop : Iterable of str
        List of column names to be removed from the DataFrame.

    Methods:
    --------
    fit(X, y)
        Fit the feature remover to the input data.

    transform(X, y=None)
        Transform the input data by removing specified features.

    Returns:
    --------
    X : pl.DataFrame
        Transformed Polars DataFrame with specified features removed.
    """

    def __init__(self, feats_to_drop: Iterable[str] = []) -> None:
        """
        Initialize the FeatureRemover.

        Parameters:
        -----------
        feats_to_drop : Iterable of str, optional (default=[])
            List of column names to be removed from the DataFrame.

        Returns:
        --------
        None
        """
        self.feats_to_drop = feats_to_drop

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """
        Fit the feature remover to the input data.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data (DataFrame).
        y : pl.Series
            Target data (Series).

        Returns:
        --------
        self : FeatureRemover
            The fitted feature remover instance.
        """
        return self

    def transform(self, X: pl.DataFrame, y=None):
        """
        Transform the input data by removing specified features.

        Parameters:
        -----------
        X : pl.DataFrame
            Input data (DataFrame) to have specified features removed.
        y : None
            Ignored. It is not used in the transformation process.

        Returns:
        --------
        X : pl.DataFrame
            Transformed Polars DataFrame with specified features removed.
        """
        X = X.drop(columns=self.feats_to_drop)
        return X


class RoundToRangeTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer for rounding values to the nearest integer within
    a specified range.

    This transformer rounds numerical values to the nearest integer
    within a specified range defined by `min_value` and `max_value`.

    Parameters:
    -----------
    min_value : int, optional (default=0)
        The minimum value of the specified range.
    max_value : int, optional (default=1)
        The maximum value of the specified range.

    Attributes:
    -----------
    min_value : int
        The minimum value of the specified range.
    max_value : int
        The maximum value of the specified range.

    Methods:
    --------
    fit(X, y)
        Fit the transformer to the input data.

    predict(X)
        Round input values to the nearest integer within the specified range.

    Returns:
    --------
    rounded_values : ndarray
        Rounded values within the specified range.
    """

    def __init__(self, min_value=0, max_value=1):
        """
        Initialize the RoundToRangeTransformer.

        Parameters:
        -----------
        min_value : int, optional (default=0)
            The minimum value of the specified range.
        max_value : int, optional (default=1)
            The maximum value of the specified range.

        Returns:
        --------
        None
        """
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None):
        """
        Fit the transformer to the input data.

        Parameters:
        -----------
        X : array-like
            Input data to be used for fitting.
        y : None
            Ignored. It is not used in the fitting process.

        Returns:
        --------
        self : RoundToRangeTransformer
            The fitted transformer instance.
        """
        return self

    def predict(self, X):
        """
        Round input values to the nearest integer within the specified range.

        Parameters:
        -----------
        X : array-like
            Input values to be rounded.

        Returns:
        --------
        rounded_values : ndarray
            Rounded values within the specified range.
        """
        # Round values to the nearest integer within the specified range
        return np.clip(np.round(X), self.min_value, self.max_value).astype(int)


class ModelWrapper(BaseEstimator, TransformerMixin):
    """
    Transformer for wrapping a machine learning model to perform
    fitting and prediction.

    This transformer wraps a machine learning model, allowing it to be used in
    a scikit-learn pipeline as an intermediate step. It performs fitting during
    the `fit` step and returns predictions during the `transform` step.

    Parameters:
    -----------
    model : object
        The machine learning model to be wrapped.
    param_grid : dict, optional (default={})
        Parameters to set for the model during fitting.

    Attributes:
    -----------
    model : object
        The wrapped machine learning model.
    param_grid : dict
        Parameters to set for the model during fitting.

    Methods:
    --------
    fit(X, y)
        Fit the model during the fit step.

    transform(X)
        Return predictions during the transform step.

    Returns:
    --------
    predictions : array-like
        Predictions made by the wrapped model.
    """

    def __init__(self, model, param_grid={}):
        """
        Initialize the ModelWrapper.

        Parameters:
        -----------
        model : object
            The machine learning model to be wrapped.
        param_grid : dict, optional (default={})
            Parameters to set for the model during fitting.

        Returns:
        --------
        None
        """
        self.model = model
        self.param_grid = param_grid

    def fit(self, X, y):
        """
        Fit the model during the fit step.

        Parameters:
        -----------
        X : array-like
            Input features for model fitting.
        y : array-like
            Target values for model fitting.

        Returns:
        --------
        self : ModelWrapper
            The fitted transformer instance.
        """
        # Fit the model during the fit step
        self.model.set_params(**self.param_grid)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        """
        Return predictions during the transform step.

        Parameters:
        -----------
        X : array-like
            Input features for making predictions.

        Returns:
        --------
        predictions : array-like
            Predictions made by the wrapped model.
        """
        # Return predictions during the transform step
        predictions = self.model.predict(X)
        return predictions
