from sklearn.base import BaseEstimator, TransformerMixin  # type:ignore
from typing import Iterable, Optional, Union
import polars as pl
import numpy as np
from boruta import BorutaPy
from collections import OrderedDict


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
        def __init__(self, name, transformer, col) -> None:
            self.transformer = transformer
            self.col = col
            self.name = name

        def fit(self, X, y=None):
            self.transformer.fit(X, y)
            return self

        def transform(self, X):
            return self.transformer.transform(X)

    def __init__(self, steps: Iterable[Step], step_params={}):
        self.steps = OrderedDict()
        for step in steps:
            self.steps[step.name] = step
        self.step_params = step_params

    def fit(self, X: Union[pl.Series, pl.DataFrame], y=None):
        if self.step_params:
            for id, params in self.step_params.items():
                self.steps[id].transformer.set_params(**params)

        for step in self.steps.values():
            step.fit(X[step.col], y)
        return self

    def transform(self, X: Union[pl.Series, pl.DataFrame], y=None):
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
    def __init__(self, drop: bool = False) -> None:
        self.categories: list
        self.cats_not_in_transform: list
        self.drop = drop

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
        X = X.select(sorted(X.columns))
        if self.drop:
            X = X[:, ::2]
        return X


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


class PolarsNullImputer(BaseEstimator, TransformerMixin):
    def __init__(self, fill_value: list) -> None:
        self.fill_value = fill_value

    def fit(self, X, y=None):
        return self

    def transform(self, X: pl.DataFrame, y=None):
        if not isinstance(X, pl.DataFrame):
            X = pl.DataFrame(X)
        bool_cols = X.select(pl.col(pl.Boolean)).columns
        for col in bool_cols:
            X = X.with_columns(pl.col(col).cast(pl.Int32).alias(col))
        X = X.fill_null(self.fill_value)
        return X


# Redefining for boruta compatibility
np.int = int
np.float = float
np.bool = bool


class BorutaFeatureSelectorPolars(BaseEstimator, TransformerMixin):
    """
    Transformer for feature selection using Boruta algorithm.

    Boruta is a feature selection algorithm that works well with ensemble methods.
    It helps identify relevant features by comparing the importance of features with
    a shadow (random) set of features. Features that are more important than the
    shadow features are selected.

    Parameters:
    -----------
    estimator : object
        The estimator to be used for feature importance comparison.
    n_estimators : int or 'auto', optional (default='auto')
        The number of base estimators to use in the ensemble. 'auto' sets it to 100.
    verbose : int, optional (default=0)
        Verbosity level.
    random_state : int or None, optional (default=None)
        Seed for random number generation.
    perc : int, optional (default=100)
        The percentile at which to stop the feature selection process.
    max_iter : int, optional (default=100)
        The maximum number of iterations to run the Boruta algorithm.
    alpha : float, optional (default=0.05)
        The significance level for testing the feature importance.
    two_step : bool, optional (default=False)
        Whether to use the two-step feature selection process.
    apply : bool, optional (default=True)
        Whether to apply feature selection or keep all features.

    Attributes:
    -----------
    selected_features : pd.Index
        Index of selected features.

    Methods:
    --------
    fit(X, y)
        Fit the Boruta feature selector to the input data.

    transform(X)
        Transform the input data by selecting relevant features.

    Returns:
    --------
    X_selected : pd.DataFrame
        A DataFrame with selected features if apply is True; otherwise,
        it returns the input data.
    """

    def __init__(
        self,
        estimator,
        n_estimators="auto",
        verbose=0,
        random_state=None,
        perc=100,
        max_iter=100,
        alpha=0.05,
        two_step=False,
        apply=True,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.random_state = random_state
        self.perc = perc
        self.max_iter = max_iter
        self.alpha = alpha
        self.two_step = two_step
        self.apply = apply

    def fit(self, X: pl.DataFrame, y: pl.Series):
        """
        Fit the Boruta feature selector to the input data.

        Parameters:
        -----------
        X : pd.DataFrame
            The feature matrix.
        y : pd.Series
            The target variable.

        Returns:
        --------
        self : BorutaFeatureSelector
            The fitted feature selector instance.
        """
        # Initialize the BorutaPy instance
        if self.apply:
            self.boruta = BorutaPy(
                estimator=self.estimator,
                n_estimators=self.estimator.n_estimators,
                verbose=self.verbose,
                random_state=self.random_state,
                perc=self.perc,
                max_iter=self.max_iter,
                alpha=self.alpha,
                two_step=self.two_step,
            )

            # Fit BorutaPy to the data
            self.boruta.fit(X.to_numpy(), y)
            self.selected_features = np.array(X.columns)[self.boruta.support_]
        return self

    def transform(self, X: pl.DataFrame):
        """
        Transform the input data by selecting relevant features.

        Parameters:
        -----------
        X : pd.DataFrame
            The feature matrix.

        Returns:
        --------
        X_selected : pd.DataFrame
            A DataFrame with selected features if apply is True; otherwise,
            it returns the input data.
        """
        # Transform the data to select the relevant features
        if self.apply:
            return X[self.selected_features]
        else:
            return X


class TargetMeanOrderedLabeler(BaseEstimator, TransformerMixin):
    def __init__(self, how: str = "mean") -> None:
        self.map = {}
        self.how = how

    def fit(self, X: pl.Series, y: pl.Series):
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
        X = X.map_dict(self.map)
        return X


class FeatureRemover(BaseEstimator, TransformerMixin):
    def __init__(self, feats_to_drop: Iterable[str] = []) -> None:
        self.feats_to_drop = feats_to_drop

    def fit(self, X: pl.DataFrame, y: pl.Series):
        return self

    def transform(self, X: pl.DataFrame, y=None):
        X = X.drop(columns=self.feats_to_drop)
        return X


class RoundToRangeTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_value=0, max_value=1):
        self.min_value = min_value
        self.max_value = max_value

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        # Round values to the nearest integer within the specified range
        return np.clip(np.round(X), self.min_value, self.max_value).astype(int)


class ModelWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, model, param_grid={}):
        self.model = model
        self.param_grid = param_grid

    def fit(self, X, y):
        # Fit the model during the fit step
        self.model.set_params(**self.param_grid)
        self.model.fit(X, y)
        return self

    def transform(self, X):
        # Return predictions during the transform step
        predictions = self.model.predict(X)
        return predictions
