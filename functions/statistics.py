import polars as pl
import scipy.stats as stats
from typing import Optional, Tuple
from statsmodels.stats.outliers_influence import variance_inflation_factor


def kruskal_polars(
    data: pl.DataFrame,
    class_col: str,
    var_col: str,
    sample_size: Optional[int] = None,
    seed: int = 1,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Perform the Kruskal-Wallis test on a Polars DataFrame.

    Args:
        data (pl.DataFrame): The Polars DataFrame containing the data.
        class_col (str): The name of the column containing class or group
        information.
        var_col (str): The name of the column containing the
        variable of interest.
        sample_size (int, optional): The sample size for each class.
        If None, use all available data for each class.
        seed (int): Seed for random sampling (if sample_size is provided).

    Returns:
        Tuple[float, float]: The Kruskal-Wallis H-statistic
        and the associated p-value.
    """
    # Get unique class values and sort them
    unique_classes = data[class_col].unique().sort()

    samples = []
    # Sample and collect data for each class
    for class_val in unique_classes:
        class_data = (
            data.filter(data[class_col] == class_val).select(var_col).drop_nulls()
        )
        if sample_size is not None:
            class_data = class_data.sample(sample_size, seed=seed)
        samples.append(class_data[var_col])

    # Perform the Kruskal-Wallis test
    result = {}
    result["statistic"], result["p_value"] = stats.kruskal(*samples)
    if result["p_value"] < alpha:
        result["message"] = (
            f"Reject the null hypothesis: There are significant "
            f"differences in '{var_col}' between groups in '{class_col}'."
        )
    else:
        result["message"] = (
            f"Fail to reject the null hypothesis: There are no "
            f"significant differences in '{var_col}' between groups "
            f"in '{class_col}'."
        )
    return result


def chi_squared_polars(
    data: pl.DataFrame, class_col: str, var_col: str, alpha: float = 0.05
):
    """
    Perform a chi-squared test of independence on a Polars DataFrame.

    Args:
        data (pl.DataFrame): The Polars DataFrame containing the data.
        class_col (str): The name of the column containing class or group information.
        var_col (str): The name of the column to be used for the test (e.g., "disbursement_method").
        alpha (float): The significance level for hypothesis testing.

    Returns:
        dict: A dictionary containing the p-value, test statistic, and result message.
    """
    data = data.select([class_col, var_col]).drop_nulls()
    contingency_table = data[var_col].value_counts()
    for cat in data[class_col].unique():
        contingency_table = contingency_table.join(
            data.filter(data[class_col] == cat)[var_col].value_counts(),
            on=var_col,
            how="left",
            suffix=f"_{cat}",
        )

    contingency_table = contingency_table.drop(columns=["count", var_col])
    contingency_table = contingency_table.drop_nulls()
    test = stats.chi2_contingency(contingency_table)

    result = {}
    result["p_value"] = test.pvalue
    result["statistic"] = test.statistic

    if result["p_value"] < alpha:
        result["message"] = (
            f"Reject the null hypothesis: There are significant "
            f"differences in '{var_col}' between groups in '{class_col}'."
        )
    else:
        result["message"] = (
            f"Fail to reject the null hypothesis: There are no "
            f"significant differences in '{var_col}' between groups "
            f"in '{class_col}'."
        )

    return result


def calc_vif(X):
    # Calculating VIF
    vif = pl.DataFrame()
    vif = vif.with_columns(pl.Series(X.columns).alias("variables"))
    vif = vif.with_columns(
        pl.Series(
            [variance_inflation_factor(X.to_numpy(), i) for i in range(X.shape[1])]
        ).alias("VIF")
    )

    return vif.sort("VIF", descending=True)


def oversample_polars(
    data: pl.DataFrame(),
    target_column,
    min_value,
):
    over_sample = pl.DataFrame()
    value_counts = data[target_column].value_counts()
    for value, count in zip(value_counts[target_column], value_counts["counts"]):
        if count < min_value:
            for i in range(int(min_value / count)):
                over_sample = over_sample.vstack(
                    data.filter(data[target_column] == value)
                )
    return over_sample
