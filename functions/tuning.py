from ray import tune, train
from sklearn.metrics import f1_score, mean_squared_error
from typing import Optional, Dict, Union
import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from ray.tune.search.optuna import OptunaSearch
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from ray.tune.stopper import (
    CombinedStopper,
    MaximumIterationStopper,
    TrialPlateauStopper,
)


def objective(
    pipeline: Pipeline,
    params: dict,
    X_train: pl.DataFrame,
    y_train: pl.Series,
    X_val: pl.DataFrame,
    y_val: pl.Series,
    n: int,
    average: str = "binary",
) -> float:
    """
    Objective function for hyperparameter tuning.

    Args:
    pipeline (Pipeline): The machine learning pipeline to be configured and
    evaluated.
    params (dict): Hyperparameter configuration for the pipeline.
    X_train (pl.DataFrame): Training data features as a Polars DataFrame.
    y_train (pl.Series): Training data labels as a Polars Series.
    X_val (pl.DataFrame): Validation data features as a Polars DataFrame.
    y_val (pl.Series): Validation data labels as a Polars Series.
    n (int): The current iteration number.
    average (str, optional): The averaging strategy for the F1 score, i.e.
    "binary", "macro", "weighted" if the value is "mse". The mean squared error
    is used instead.

    Returns:
    float: The F1 score (or negative mean squared error if average is "mse")
    as the objective to be optimized.
    """
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    if average != "mse":
        score = f1_score(y_val, preds, average=average)
    if average == "mse":
        score = -mean_squared_error(y_val, preds)
    print(f"Step {n} F-1 Score: {score}")
    return score


class Trainable(tune.Trainable):
    """
    A custom Ray Tune trainable class for hyperparameter tuning.

    This class is used to configure and execute hyperparameter tuning experiments
    using Ray Tune. It sets up the necessary parameters and data for each trial,
    and performs steps to evaluate the hyperparameter configurations.

    Attributes:
    - config (dict): A dictionary of hyperparameters for the pipeline.
    - pipeline: The machine learning pipeline to be configured and evaluated.
    - X_train: Training data features.
    - y_train: Training data labels.
    - X_val: Validation data features.
    - y_val: Validation data labels.
    - sample_size (Union[int, str]): The sample size for data splitting.
    - average (str): The averaging strategy for the F1 score, either "binary" or "mse".
    - stratify (bool): Whether to stratify data splitting.

    Methods:
    - setup(config, pipeline, X_train, y_train, X_val, y_val, sample_size, average, stratify):
        Set up the trainable object with hyperparameters and data.

    - step():
        Perform a training step and return the score.

    """

    def setup(
        self,
        config: dict,
        pipeline: Pipeline,
        X_train: pl.DataFrame,
        y_train: pl.Series,
        X_val: pl.DataFrame,
        y_val: pl.Series,
        sample_size: Union[int, str] = 100000,
        average: str = "binary",
        stratify: bool = True,
    ):
        """
        Set up the trainable object with hyperparameters and data.

        Args:
        config (dict): A dictionary of hyperparameters.
        pipeline: The machine learning pipeline.
        X_train: Training data features.
        y_train: Training data labels.
        X_val: Validation data features.
        y_val: Validation data labels.
        sample_size (Union[int, str], optional): The sample size for data
        splitting. Default is 100,000.

        average (str, optional): The averaging strategy for the F1 score,
        either "binary" or "mse". Default is "binary".

        stratify (bool, optional): Whether to stratify data splitting.
        Default is True.
        """
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.params = config
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.sample_size = sample_size
        self.average = average

        if self.sample_size != "all":
            if stratify:
                self.splitter_train = StratifiedShuffleSplit(
                    n_splits=10,
                    random_state=1,
                    test_size=100,
                    train_size=self.sample_size,
                )
                self.splitter_test = StratifiedShuffleSplit(
                    n_splits=10,
                    random_state=1,
                    test_size=100,
                    train_size=int(self.sample_size / 3),
                )
            else:
                self.splitter_train = ShuffleSplit(
                    n_splits=10,
                    random_state=1,
                    test_size=100,
                    train_size=self.sample_size,
                )
                self.splitter_test = ShuffleSplit(
                    n_splits=10,
                    random_state=1,
                    test_size=100,
                    train_size=int(self.sample_size / 3),
                )

            self.split_train = self.splitter_train.split(self.X_train, self.y_train)
            self.split_test = self.splitter_test.split(self.X_val, self.y_val)

            self.split_idx_train = {}
            for i, split_id in enumerate(self.split_train):
                self.split_idx_train[i] = split_id[0]

            self.split_idx_test = {}
            for i, split_id in enumerate(self.split_test):
                self.split_idx_test[i] = split_id[0]

    def step(self):
        """
        Perform a training step.

        Returns:
        dict: A dictionary containing the score for the current step.
        """
        if self.sample_size != "all":
            X_train = self.X_train[self.split_idx_train[self.x]]
            y_train = self.y_train[self.split_idx_train[self.x]]
            X_val = self.X_val[self.split_idx_test[self.x]]
            y_val = self.y_val[self.split_idx_test[self.x]]
        else:
            X_train = self.X_train
            y_train = self.y_train
            X_val = self.X_val
            y_val = self.y_val
        score = objective(
            self.pipeline,
            self.params,
            X_train,
            y_train,
            X_val,
            y_val,
            self.x,
            self.average,
        )
        self.x += 1
        return {"score": score}


class Models:
    """
    Container for managing and evaluating machine learning models using Polars
    and Ray Tune for hyperparameter optimization.

    This class allows you to add, tune, and evaluate machine learning models.

    Attributes:
    -----------
    models : dict
        A dictionary to store machine learning models.

    Methods:
    --------
    add_model(model_name, pipeline, param_grid, override_n=None,
    metric_threshold=0.55)
        Add a machine learning model to the container.

    remove_model(model_name)
        Remove a machine learning model from the container.

    tune_all(X_train, y_train, X_val, y_val, **kwargs)
        Tune and cross-validate all models in the container.

    """

    def __init__(self) -> None:
        self.models: dict = {}

    class Model:
        """
        Represents an individual machine learning model with methods for
        tuning and evaluation.

        Parameters:
        -----------
        name : str
            The name of the model.
        pipeline : Pipeline
            The scikit-learn pipeline for the model.
        param_grid : dict
            A dictionary of hyperparameter search spaces for the model.
        metric_threshold : float, optional
            Threshold of tuning metric for early stopping. Default is 0.55.
        override_n : int, optional
            Number of trials for hyperparameter optimization.
            Overrides the default value.

        Methods:
        --------
        tune_model(X_train, y_train, X_val, y_val, n=10, n_training=10,
        sample_size=100000, average="binary", stratify=True)
            Tune the model's hyperparameters using Ray Tune and Optuna.

        """

        def __init__(
            self,
            name: str,
            pipeline: Pipeline,
            param_grid: Dict,
            metric_threshold: float = 0.55,
            override_n: Optional[int] = None,
        ) -> None:
            self.pipeline: Pipeline = pipeline
            self.param_grid = param_grid
            self.best_params: Dict
            self.override_n = override_n
            self.name = name
            self.metric_threshold = metric_threshold

        def tune_model(
            self,
            X_train: Union[pl.DataFrame, np.ndarray, pl.Series],
            y_train: Union[pl.DataFrame, np.ndarray, pl.Series],
            X_val: Union[pl.DataFrame, np.ndarray, pl.Series],
            y_val: Union[pl.DataFrame, np.ndarray, pl.Series],
            n: int = 10,
            n_training: int = 10,
            sample_size: int = 100000,
            average: str = "binary",
            stratify=True,
        ):
            """
            Tune the model's hyperparameters using Ray Tune and Optuna.

            Parameters:
            X_train : Union[pl.DataFrame, np.ndarray, pl.Series]
                The feature matrix for training.
            y_train : Union[pl.DataFrame, np.ndarray, pl.Series]
                The target variable for training.
            X_val : Union[pl.DataFrame, np.ndarray, pl.Series]
                The feature matrix for validation.
            y_val : Union[pl.DataFrame, np.ndarray, pl.Series]
                The target variable for validation.
            n : int, optional
                Number of trials for hyperparameter optimization.
                Default is 10.
            n_training : int, optional
                Number of training iterations. Default is 10.
            sample_size : int, optional
                The sample size for data splitting. Default is 100,000.
            average : str, optional
                The averaging strategy for the F1 score, either "binary",
                "macro", "average" or "mse" if mean squared error is to
                be used instead. Default is "binary".
            stratify : bool, optional
                Whether to stratify data splitting. Default is True.
            """
            stopper = CombinedStopper(
                MaximumIterationStopper(n_training),
                TrialPlateauStopper(
                    std=0.2,
                    grace_period=1,
                    num_results=3,
                    metric="score",
                    metric_threshold=self.metric_threshold,
                    mode="min",
                ),
            )
            tuner = tune.Tuner(
                trainable=tune.with_resources(
                    tune.with_parameters(
                        Trainable,
                        pipeline=self.pipeline,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        sample_size=sample_size,
                        average=average,
                        stratify=stratify,
                    ),
                    resources={"CPU": 2},
                ),
                run_config=train.RunConfig(
                    stop=stopper,
                    storage_path=f"/tmp/tune_results/",
                    name=self.name,
                    checkpoint_config=train.CheckpointConfig(checkpoint_at_end=False),
                ),
                tune_config=tune.TuneConfig(
                    search_alg=OptunaSearch(), mode="max", metric="score", num_samples=n
                ),
                param_space=self.param_grid,
            )

            results = tuner.fit()
            self.best_params = results.get_best_result().config
            self.pipeline.set_params(**self.best_params)

    def add_model(
        self,
        model_name: str,
        pipeline: Pipeline,
        param_grid: Dict,
        override_n: Optional[int] = None,
        metric_threshold: float = 0.55,
    ):
        """
        Add a machine learning model to the container.

        Parameters:
        model_name : str
            The name of the model.
        pipeline : Pipeline
            The scikit-learn pipeline for the model.
        param_grid : Dict
            A dictionary of hyperparameter search spaces for the model.
        override_n : int, optional
            Number of trials for hyperparameter optimization.
            Overrides the default value.
        metric_threshold : float, optional
            Threshold of tuning metric for early stopping. Default is 0.55.
        """
        self.models[model_name] = self.Model(
            model_name,
            pipeline,
            param_grid,
            override_n=override_n,
            metric_threshold=metric_threshold,
        )

    def remove_model(self, model_name: str):
        """
        Remove a machine learning model from the container.

        Parameters:
            model_name : str
                The name of the model to be removed.
        """
        if model_name in self.models:
            del self.models[model_name]

    def tune_all(
        self,
        X_train: Union[pl.DataFrame, np.ndarray, pl.Series],
        y_train: Union[pl.DataFrame, np.ndarray, pl.Series],
        X_val: Union[pl.DataFrame, np.ndarray, pl.Series],
        y_val: Union[pl.DataFrame, np.ndarray, pl.Series],
        **kwargs,
    ):
        """
        Tune and cross-validate all models in the container.

        Parameters:
        X_train : Union[pl.DataFrame, np.ndarray, pl.Series]
            The feature matrix for training.
        y_train : Union[pl.DataFrame, np.ndarray, pl.Series]
            The target variable for training.
        X_val : Union[pl.DataFrame, np.ndarray, pl.Series]
            The feature matrix for validation.
        y_val : Union[pl.DataFrame, np.ndarray, pl.Series]
            The target variable for validation.
        **kwargs
            Additional keyword arguments to be passed to the tune_model method.
        """
        for name, model in self.models.items():
            if model.override_n:
                kwargs["n"] = model.override_n
            model.tune_model(X_train, y_train, X_val, y_val, **kwargs)
            print(f"{name} tuned.")
