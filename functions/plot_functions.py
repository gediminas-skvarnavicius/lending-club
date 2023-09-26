import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_x_axis_break(ax1: plt.Axes, ax2: plt.Axes, d: float = 1.5, **kwargs) -> None:
    """
    Plot an x-axis break on two given axes.

    Parameters:
        ax1 (matplotlib.axes.Axes): The first axis on which to plot the x-axis break.
        ax2 (matplotlib.axes.Axes): The second axis on which to plot the x-axis break.
        d (float): Proportion of vertical to horizontal extent of the slanted line.
        **kwargs: Additional keyword arguments for customizing the appearance of the break.

    Example usage:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        plot_x_axis_break(ax1, ax2, color="k", mec="k", mew=1, markersize=12)
    """
    # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(
        marker=[(-1, -d), (1, d)],
        markersize=12,
        linestyle="none",
        color="k",
        mec="k",
        mew=1,
        clip_on=False,
    )
    ax1.plot([1], [0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0], [0], transform=ax2.transAxes, **kwargs)


import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def plot_grouped_bars(
    df: pl.DataFrame,
    category_col: str = None,
    col1: str = None,
    col2: str = None,
    group_width: float = 0.4,
    bar_width: float = 0.4,
    top_vals: int = None,
    return_fig: bool = False,
    figsize_args: dict = {},  # Dictionary for figsize arguments
    legend_args: dict = {},  # Dictionary for legend arguments
    x_label_args: dict = {},
):
    """
    Create a grouped bar plot with two y-axes for two columns in a Polars DataFrame.

    Parameters:
        df (pl.DataFrame): The input Polars DataFrame.
        category_col (str, optional): The column for grouping the bars. Defaults to the first column in the DataFrame.
        col1 (str, optional): The first column to plot on the left y-axis. Defaults to the second column in the DataFrame.
        col2 (str, optional): The second column to plot on the right y-axis. Defaults to the third column in the DataFrame.
        group_width (float, optional): Width of each group of bars. Defaults to 0.4.
        bar_width (float, optional): Width of each individual bar within a group. Defaults to 0.4.
        top_vals (int, optional): Display only the top N values in the category_col based on col1. Defaults to None (display all values).
        return_fig (bool, optional): If True, returns the Matplotlib figure and axes. Defaults to False.
        **kwargs: Additional keyword arguments for figure and legend customization.

    Returns:
        None: If return_fig is False (default).
        Tuple[matplotlib.figure.Figure, Tuple[matplotlib.axes._subplots.AxesSubplot]]: If return_fig is True.
    """
    if not category_col:
        category_col = df.columns[0]
    if not col1:
        col1 = df.columns[1]
    if not col2:
        col2 = df.columns[2]

    if top_vals:
        mask = df[category_col].is_in(df.sort(col1)[-top_vals:][category_col])
        df = df.filter(mask)

    df = df.sort(category_col, descending=True)
    categories = df[category_col].unique().sort(descending=True)

    x_positions = np.arange(len(categories))
    x_offsets = [-group_width / 2, group_width / 2]
    x1_positions = x_positions + x_offsets[0]
    x2_positions = x_positions + x_offsets[1]

    fig, ax1 = plt.subplots(**figsize_args)
    ax2 = ax1.twinx()

    bar1 = ax1.bar(
        x=x1_positions,
        height=df[col1],
        width=bar_width,
        label=col1,
    )

    bar2 = ax2.bar(
        x=x2_positions,
        height=df[col2],
        width=bar_width,
        color="orange",
        label=col2,
    )

    ax1.set_xticks(np.arange(len(categories)))
    ax1.set_xticklabels(categories, **x_label_args)

    ax1.set_ylabel(col1)
    ax2.set_ylabel(col2)
    ax2.legend([bar1, bar2], [bar1.get_label(), bar2.get_label()], **legend_args)

    ax1.grid(None)
    ax2.grid(None)

    plt.tight_layout()

    if return_fig:
        return fig, (ax1, ax2)
    else:
        plt.show()
