from ray import tune, train
from sklearn.metrics import f1_score
from typing import Optional, Dict, Union
import polars as pl
import numpy as np
from sklearn.pipeline import Pipeline
from ray.tune.search.optuna import OptunaSearch


def objective(pipeline, params, X_train, y_train, X_val, y_val, n):
    pipeline.set_params(**params)
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_val)
    f1 = f1_score(y_val, preds)
    print(f"Step {n} F-1 Score: {f1}")
    return f1


class Trainable(tune.Trainable):
    def setup(self, config: dict, pipeline, X_train, y_train, X_val, y_val):
        # config (dict): A dict of hyperparameters
        self.x = 0
        self.params = config
        self.pipeline = pipeline
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def step(self):  # This is called iteratively.
        print(self.get_config())
        score = objective(
            self.pipeline,
            self.params,
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.x,
        )
        self.x += 1
        return {"score": score}


class Models:
    """
    Container for managing and evaluating machine learning models.

    This class allows you to add, tune, and evaluate machine learning models
    using Optuna for hyperparameter optimization.

    Attributes:
    -----------
    models : dict
        A dictionary to store machine learning models.

    Methods:
    --------
    add_model(model_name, pipeline, param_grid, override_n=None)
        Add a machine learning model to the container.

    remove_model(model_name)
        Remove a machine learning model from the container.

    tune_val_all(X, y, score="f1", n=100)
        Tune and cross-validate all models in the container.

    Inner Class:
    ------------
    Model
        Represents an individual machine learning model with methods for tuning and evaluation.

        Methods:
        --------
        tune_model(X, y, n=100, starting_params=None)
            Tune the model's hyperparameters using Optuna.

        cross_val(X, y, n=10, random_state=1, score="f1")
            Perform k-fold cross-validation for the model.

        calc_perm_importances(X, y, score="f1", n=10, random_seed=1)
            Calculate feature importances using permutation importance.

    """

    def __init__(self) -> None:
        self.models: dict = {}

    class Model:
        """
        Represents an individual machine learning model with methods for tuning and evaluation.

        Parameters:
        -----------
        pipeline : Pipeline
            The scikit-learn pipeline for the model.
        param_grid : dict
            A dictionary of hyperparameter search spaces for the model.
        override_n : int, optional
            Number of trials for hyperparameter optimization. Overrides the default value.

        Methods:
        --------
        tune_model(X, y, n=100, starting_params=None)
            Tune the model's hyperparameters using Optuna.

        cross_val(X, y, n=10, random_state=1, score="f1")
            Perform k-fold cross-validation for the model.

        calc_perm_importances(X, y, score="f1", n=10, random_seed=1)
            Calculate feature importances using permutation importance.
        """

        def __init__(
            self,
            name: str,
            pipeline: Pipeline,
            param_grid: Dict,
            override_n: Optional[int] = None,
        ) -> None:
            self.pipeline: Pipeline = pipeline
            self.param_grid = param_grid
            self.best_params: Dict
            self.override_n = override_n
            self.name = name

        def tune_model(
            self,
            X_train: Union[pl.DataFrame, np.ndarray, pl.Series],
            y_train: Union[pl.DataFrame, np.ndarray, pl.Series],
            X_val: Union[pl.DataFrame, np.ndarray, pl.Series],
            y_val: Union[pl.DataFrame, np.ndarray, pl.Series],
            n: int = 10,
            n_training: int = 5,
            starting_params: Optional[Dict] = None,
        ):
            """
            Tune the model's hyperparameters using Optuna.

            Parameters:
                X : pd.DataFrame or pd.Series
                    The feature matrix.
                y : pd.Series or np.ndarray
                    The target variable.
                n : int, optional
                    Number of trials for hyperparameter optimization. Default is 100.
                starting_params : dict, optional
                    Starting hyperparameters for optimization. Default is None.
            """
            tuner = tune.Tuner(
                trainable=tune.with_resources(
                    tune.with_parameters(
                        Trainable,
                        pipeline=self.pipeline,
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                    ),
                    resources={"CPU": 1},
                ),
                run_config=train.RunConfig(
                    stop={"training_iteration": n_training},
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

        # def cross_val(
        #     self,
        #     X: Union[pd.DataFrame, pd.Series],
        #     y: Union[pd.Series, np.ndarray],
        #     n: int = 10,
        #     random_state: int = 1,
        #     score: str = "f1",
        # ):
        #     """
        #     Perform k-fold cross-validation for the model.

        #     Parameters:
        #         X : pd.DataFrame or pd.Series
        #             The feature matrix.
        #         y : pd.Series or np.ndarray
        #             The target variable.
        #         n : int, optional
        #             Number of cross-validation folds. Default is 10.
        #         random_state : int, optional
        #             Random seed for shuffling data during cross-validation. Default is 1.
        #         score : str, optional
        #             Scoring metric for cross-validation. Default is "f1".
        #     """
        #     kf = StratifiedKFold(n_splits=n, shuffle=True, random_state=random_state)
        #     self.cv_results = cross_val_score(
        #         self.pipeline, X, y, cv=kf, scoring=score, n_jobs=2
        #     )

        # def calc_perm_importances(
        #     self,
        #     X: Union[pd.DataFrame, pd.Series],
        #     y: Union[pd.Series, np.ndarray],
        #     score: str = "f1",
        #     n: int = 10,
        #     random_seed: int = 1,
        # ):
        #     """
        #     Calculate feature importances using permutation importance.

        #     Parameters:
        #         X : pd.DataFrame or pd.Series
        #             The feature matrix.
        #         y : pd.Series or np.ndarray
        #             The target variable.
        #         score : str, optional
        #             Scoring metric for permutation importance. Default is "f1".
        #         n : int, optional
        #             Number of permutation iterations. Default is 10.
        #         random_seed : int, optional
        #             Random seed for reproducibility. Default is 1.
        #     """
        #     self.permut_importances = permutation_importance(
        #         self.pipeline["model"],
        #         self.pipeline["preprocess"].transform(X),
        #         y,
        #         scoring=score,
        #         n_repeats=n,
        #         random_state=random_seed,
        #     )

    def add_model(
        self,
        model_name: str,
        pipeline: Pipeline,
        param_grid: Dict,
        override_n: Optional[int] = None,
    ):
        """
        Add a machine learning model to the container.

        Parameters:
            model_name : str
                The name of the model.
            pipeline : Pipeline
                The scikit-learn pipeline for the model.
            param_grid : dict
                A dictionary of hyperparameter search spaces for the model.
            override_n : int, optional
                Number of trials for hyperparameter optimization. Overrides the default value.
        """
        self.models[model_name] = self.Model(
            model_name, pipeline, param_grid, override_n=override_n
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
        score: str = "f1",
        n: int = 10,
    ):
        """
        Tune and cross-validate all models in the container.

        Parameters:
            X : pd.DataFrame
                The feature matrix.
            y : pd.Series or np.ndarray
                The target variable.
            score : str, optional
                Scoring metric for cross-validation. Default is "f1".
            n : int, optional
                Number of trials for hyperparameter optimization. Default is 100.
        """
        for name, model in self.models.items():
            if model.override_n:
                model.tune_model(X_train, y_train, X_val, y_val, n=model.override_n)
            else:
                model.tune_model(X_train, y_train, X_val, y_val, n=n)
            print(f"{name} tuned:")
            # print("Best Trial:")
            # print(f"  Value: {model.study.best_trial.value}")
            # print("  Params: ")
            # for key, value in model.study.best_trial.params.items():
            #     print(f"    {key}: {value}")
            # model.cross_val(X, y, score=score)
            # print(f"{score} CV mean score: {round(model.cv_results.mean(),3)}")