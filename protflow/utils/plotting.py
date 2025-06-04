"""
Plotting Module
===============

This module provides functionality for creating various standardized plots, primarily focusing on violin plots. It offers tools to generate, customize, and save plots in a structured and automated manner, making it ideal for data visualization within scientific and analytical workflows.

Detailed Description
--------------------
The `PlottingTrajectory` class encapsulates the functionality necessary to generate violin plots. It manages the configuration of plot parameters, handles the addition of data, and executes the plotting processes. The class includes methods for customizing the appearance of plots, ensuring the resulting visualizations are both informative and aesthetically pleasing. Additionally, standalone functions for creating scatter plots, sequence logos, and other specific plot types are provided.

The module is designed to streamline the creation and customization of plots, supporting automatic setup of plot parameters, execution of plotting commands, and saving of output files in various formats. This facilitates subsequent data analysis and presentation steps.

Usage
-----
To use this module, create an instance of the `PlottingTrajectory` class and invoke its methods with appropriate parameters. The module handles the configuration, execution, and result collection processes. Detailed control over the plotting process is provided through various parameters, allowing for customized visualizations tailored to specific research needs.

Examples
--------
Here is an example of how to initialize and use the `PlottingTrajectory` class:

.. code-block:: python

    from plotting import PlottingTrajectory

    # Initialize the PlottingTrajectory class
    plotter = PlottingTrajectory(y_label="Value", location="output/violin_plot.png")

    # Add data to the plot
    plotter.add([1, 2, 3, 4, 5], label="Sample 1")
    plotter.add([2, 3, 4, 5, 6], label="Sample 2")

    # Generate and save the violin plot
    plotter.violin_plot()

Standalone function usage:

.. code-block:: python

    from plotting import scatterplot

    # Create a scatter plot from a DataFrame
    scatterplot(dataframe=df, x_column='x_data', y_column='y_data', out_path='output/scatter_plot.png')

Further Details
---------------
    - Edge Cases: The module handles various edge cases, such as empty data lists and invalid plot parameters. It ensures robust error handling and logging for easier debugging and verification of the plotting process.
    - Customizability: Users can customize plots through multiple parameters, including colormap selection, axis labels, and plot dimensions.
    - Integration: The module seamlessly integrates with other components of data analysis workflows, leveraging shared configurations and data structures to provide a cohesive user experience.

This module is intended for researchers and developers who need to create detailed and customizable plots for their data analysis and presentation needs. By automating many of the setup and execution steps, it allows users to focus on interpreting results and advancing their scientific inquiries.

Notes
-----
This module is designed to work independently or in tandem with other components of data analysis packages, particularly those related to data visualization and presentation.

Authors
-------
Markus Braun, Adrian Tripp

Version
-------
1.0.0
"""
# dependencies
import logging
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import os

class PlottingTrajectory():
    def __init__(self, y_label: str, location: str, title: str = "Refinement Trajectory", dims = None):
        '''AAA'''
        self.title = title
        self.y_label = y_label
        self.location = location
        self.dims = dims
        self.data = list()
        self.colormap="tab20"

    def set_dims(self, dims):
        '''AAA'''
        self.dims = dims
        return None

    def set_y_label(self, label: str) -> None:
        self.y_label = label
        return None

    def set_location(self, location: str) -> None:
        ''''''
        self.location = location
        return None

    def set_colormap(self, colormap: str) -> None:
        self.colormap = colormap
        return None

    def add(self, data_list: list, label: str) -> None:
        '''AAA'''
        if not isinstance(data_list, list):
            data_list = list(data_list)
        self.data.append((label, data_list))
        return None

    def violin_plot(self, out_path:str=None, show_fig:bool=False):
        '''AAA'''
        out_path = out_path or self.location
        if not self.data:
            return logging.info(f"Nothing can be plotted, no data added yet.")
        def set_violinstyle(axes_subplot_parts, colors="cornflowerblue") -> None:
            '''AAA'''
            for color, pc in zip(colors, axes_subplot_parts["bodies"]):
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(1)
            axes_subplot_parts["cmins"].set_edgecolor("black")
            axes_subplot_parts["cmaxes"].set_edgecolor("black")
            return None

        # get colors from colormap
        colors = [mcolors.to_hex(color) for color in plt.get_cmap(self.colormap).colors]
        fig, ax = plt.subplots(1, 1, figsize=(3+0.8*(len(self.data)), 5), constrained_layout=True)

        #for ax, col, name, label, dim in zip(ax_list, cols, titles, y_labels, dims):
        ax.set_title(self.title, size=15, y=1.05)
        ax.set_ylabel(self.y_label, size=15)
        ax.set_xticks([])
        parts = ax.violinplot([x[1] for x in self.data], widths=0.7)
        if self.dims: ax.set_ylim(self.dims)
        set_violinstyle(parts, colors=colors)

        for i, d in enumerate([x[1] for x in self.data]):
            quartile1, median, quartile3 = np.percentile(d, [25, 50, 75])
            ax.scatter(i+1, median, marker='o', color="white", s=40, zorder=3)
            ax.vlines(i+1, quartile1, quartile3, color="k", linestyle="-", lw=5)
            ax.vlines(i+1, np.min(d), np.max(d), color="k", linestyle="-", lw=2)

        handles = [mpatches.Patch(color=c, label=f"{l[0]} (n={len(l[1])})") for c, l in zip(colors, [x for x in self.data])]
        fig.legend(
            handles=handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),  # just outside the axes to the right
            fancybox=True,
            shadow=True,
            fontsize=13,
            ncol=1  # vertical stack on the right
        )

        if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches='tight')
        if show_fig: fig.show()
        return None

    def add_and_plot(self, data_list, label, show_fig: bool=False):
        ''''''
        self.add(data_list, label)
        self.violin_plot(show_fig=show_fig)
        return None

def singular_violinplot(data: list, y_label: str, title: str, out_path: str = None, show_fig: bool=False) -> None:
    """
    Create a Singular Violin Plot
    =============================
    
    This function generates a singular violin plot from a provided list of data points. It allows for the customization of the y-axis label and plot title, and optionally saves the resulting plot to a specified file path.

    Parameters
    ----------
    data (list): 
        A list of numerical data points to be visualized in the violin plot.
    y_label (str): 
        The label for the y-axis of the plot.
    title (str): 
        The title of the plot.
    out_path (str, optional): 
        The file path where the plot should be saved. If not provided, the plot will be displayed without saving.

    Detailed Description
    --------------------
    The `singular_violinplot` function creates a violin plot to visualize the distribution of a single set of data points. The plot includes median, quartiles, and range indicators, providing a comprehensive view of the data distribution. The function leverages Matplotlib for plot creation and supports customization of several plot attributes to enhance the visual representation of the data.

    The function:
    - Generates a violin plot for the provided data.
    - Sets the plot title and y-axis label according to the provided parameters.
    - Highlights the median, quartiles, and range of the data.
    - Optionally saves the plot to the specified file path.

    Examples
    --------
    Here is an example of how to use the `singular_violinplot` function:

    .. code-block:: python

        from plotting import singular_violinplot

        # Define data
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Create a violin plot
        singular_violinplot(data=data, y_label="Value", title="Sample Violin Plot", out_path="output/violin_plot.png")

    Further Details
    ---------------
    - Edge Cases: The function handles empty data lists by not attempting to plot and logging a message.
    - Customizability: Users can customize the y-axis label, plot title, and output path for saving the plot.
    - Integration: The function can be used as part of larger data analysis workflows, integrating seamlessly with other plotting functions and data processing steps.

    This function is intended for researchers and developers who need to create detailed and customizable violin plots for their data analysis and presentation needs.

    Notes
    -----
    This function is part of the Plotting module and is designed to work in tandem with other plotting functions provided in the module.
    """
    fig, ax = plt.subplots(figsize=(2,5))

    parts = ax.violinplot(data, widths=0.5)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(y_label, size=13) # "\u00C5" is Unicode for Angstrom
    ax.set_xticks([])

    quartile1, median, quartile3 = np.percentile(data, [25, 50, 75]) #axis=1 if multiple violinplots.

    for pc in parts['bodies']:
        pc.set_facecolor('cornflowerblue')
        pc.set_edgecolor('black')
        pc.set_alpha(1)

    parts["cmins"].set_edgecolor("black")
    parts["cmaxes"].set_edgecolor("black")

    ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
    ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
    ax.vlines(1, np.min(data), np.max(data), color="k", linestyle="-", lw=2)

    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    if show_fig: fig.show()
    return None

def violinplot_multiple_cols_dfs(dfs: list[pd.DataFrame], df_names: list[str], cols: list[str], y_labels: list[str], titles: list[str] = None, dims: list[tuple[float,float]] = None, out_path: str = None, colormap: str = "tab20", show_fig: bool = True) -> None:
    """
    Create Multiple Violin Plots from DataFrame Columns
    ===================================================
    
    This function generates multiple violin plots from specified columns of multiple Pandas DataFrames. It allows for customization of axis labels, plot titles, and plot dimensions, and optionally saves the resulting plots to a specified file path.

    Parameters
    ----------
    dfs (list[pd.DataFrame]): 
        A list of Pandas DataFrames containing the data to be visualized.
    df_names (list[str]): 
        A list of names corresponding to each DataFrame, used for labeling in the legend.
    cols (list[str]): 
        A list of column names from the DataFrames to be visualized in the violin plots.
    y_labels (list[str]): 
        A list of labels for the y-axes of the plots.
    titles (list[str], optional): 
        A list of titles for each plot. If not provided, plots will not have titles.
    dims (list[tuple[float, float]], optional): 
        A list of tuples specifying the y-axis limits for each plot. If not provided, the y-axis limits will be determined automatically.
    out_path (str, optional): 
        The file path where the plots should be saved. If not provided, the plots will be displayed without saving.
    colormap (str, optional): 
        The colormap to be used for coloring the plots. Defaults to "tab20".
    show_fig (bool, optional): 
        Whether to display the plot. Defaults to True.

    Detailed Description
    --------------------
    The `violinplot_multiple_cols_dfs` function creates multiple violin plots to visualize the distributions of specified columns from multiple Pandas DataFrames. The function supports extensive customization options, including axis labels, plot titles, colormap selection, and plot dimensions. It is designed to handle the intricacies of plotting data from different DataFrames, ensuring that each plot is clearly labeled and informative.

    The function:
    - Validates the presence of specified columns in the DataFrames.
    - Generates violin plots for each specified column across the DataFrames.
    - Sets plot titles, y-axis labels, and y-axis limits according to the provided parameters.
    - Applies a consistent colormap to enhance visual distinction between different DataFrames.
    - Optionally saves the plots to the specified file path.

    Examples
    --------
    Here is an example of how to use the `violinplot_multiple_cols_dfs` function:

    .. code-block:: python

        from plotting import violinplot_multiple_cols_dfs
        import pandas as pd

        # Create sample DataFrames
        df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        df2 = pd.DataFrame({'col1': [3, 4, 5], 'col2': [6, 7, 8]})

        # Define parameters
        dfs = [df1, df2]
        df_names = ["DataFrame 1", "DataFrame 2"]
        cols = ["col1", "col2"]
        y_labels = ["Value 1", "Value 2"]
        titles = ["Plot 1", "Plot 2"]

        # Create and save the violin plots
        violinplot_multiple_cols_dfs(dfs=dfs, df_names=df_names, cols=cols, y_labels=y_labels, titles=titles, out_path="output/violin_plots.png")

    Further Details
    ---------------
    - Edge Cases: The function handles missing columns by raising a KeyError and suggesting similar column names.
    - Customizability: Users can customize the y-axis labels, plot titles, colormap, and output path for saving the plots.
    - Integration: The function can be used as part of larger data analysis workflows, integrating seamlessly with other plotting functions and data processing steps.

    This function is intended for researchers and developers who need to create detailed and customizable violin plots for comparing data across multiple DataFrames.

    Notes
    -----
    This function is part of the Plotting module and is designed to work in tandem with other plotting functions provided in the module.
    """
    # security
    for df in dfs:
        for col in cols:
            check_for_col_in_df(col, df)

    if not titles:
        titles = ["" for _ in cols]

    def set_violinstyle(axes_subplot_parts, colors="cornflowerblue") -> None:
        '''
        '''
        for color, pc in zip(colors, axes_subplot_parts["bodies"]):
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(1)
        axes_subplot_parts["cmins"].set_edgecolor("black")
        axes_subplot_parts["cmaxes"].set_edgecolor("black")
        return None

    # get colors from colormap
    colors = [mcolors.to_hex(color) for color in plt.get_cmap(colormap).colors]
    fig, ax_list = plt.subplots(1, len(cols), figsize=(3*len(cols)+0.8*(len(dfs)), 5))
    # TODO: plt.subplots returns a single, non-iterable axis object if len(cols) = 1, therefore we need to put it in a list to make it iterable. No idea why this was not the case in iterative refinement
    if not isinstance(ax_list, np.ndarray):
        ax_list = [ax_list]

    fig.subplots_adjust(wspace=1, hspace=0.8)
    if not dims: dims = [None for x in cols]

    for ax, col, name, label, dim in zip(ax_list, cols, titles, y_labels, dims):
        ax.set_title(name, size=15, y=1.05)
        ax.set_ylabel(label, size=15)

        ax.set_xticks([])
        data = [df[col].to_list() for df in dfs]
        parts = ax.violinplot([df[col].to_list() for df in dfs], widths=0.7)
        if dim: ax.set_ylim(dim)
        set_violinstyle(parts, colors=colors)

        for i, d in enumerate(data):
            quartile1, median, quartile3 = np.percentile(d, [25, 50, 75])
            ax.scatter(i+1, median, marker='o', color="white", s=40, zorder=3)
            ax.vlines(i+1, quartile1, quartile3, color="k", linestyle="-", lw=5)
            ax.vlines(i+1, np.min(d), np.max(d), color="k", linestyle="-", lw=2)

        labels = [f"{l} (n={len(df.index)})" for l, df in zip(df_names, dfs)]
        handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.1),
                  fancybox=True, shadow=True, ncol=5, fontsize=13)

    if out_path:
        fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    if show_fig:
        fig.show()
    return None

def violinplot_multiple_cols(dataframe: pd.DataFrame, cols: list[str], y_labels: list[str], titles: list[str] = None, dims: list[tuple[int,int]] = None, out_path: str = None, show_fig: bool = True) -> None:
    """
    Create Multiple Violin Plots from DataFrame Columns
    ===================================================
    
    This function generates multiple violin plots from specified columns of a single Pandas DataFrame. It allows for customization of axis labels, plot titles, and plot dimensions, and optionally saves the resulting plots to a specified file path.

    Parameters
    ----------
    dataframe (pd.DataFrame): 
        A Pandas DataFrame containing the data to be visualized.
    cols (list[str]): 
        A list of column names from the DataFrame to be visualized in the violin plots.
    y_labels (list[str]): 
        A list of labels for the y-axes of the plots.
    titles (list[str], optional): 
        A list of titles for each plot. If not provided, plots will not have titles.
    dims (list[tuple[int, int]], optional): 
        A list of tuples specifying the y-axis limits for each plot. If not provided, the y-axis limits will be determined automatically.
    out_path (str, optional): 
        The file path where the plots should be saved. If not provided, the plots will be displayed without saving.
    show_fig (bool, optional): 
        Whether to display the plot. Defaults to True.

    Detailed Description
    --------------------
    The `violinplot_multiple_cols` function creates multiple violin plots to visualize the distributions of specified columns from a single Pandas DataFrame. The function supports extensive customization options, including axis labels, plot titles, and plot dimensions. It ensures that each plot is clearly labeled and informative, providing a comprehensive view of the data distribution.

    The function:
    - Validates the presence of specified columns in the DataFrame.
    - Generates violin plots for each specified column.
    - Sets plot titles, y-axis labels, and y-axis limits according to the provided parameters.
    - Optionally saves the plots to the specified file path.

    Examples
    --------
    Here is an example of how to use the `violinplot_multiple_cols` function:

    .. code-block:: python

        from plotting import violinplot_multiple_cols
        import pandas as pd

        # Create a sample DataFrame
        df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': [7, 8, 9]})

        # Define parameters
        cols = ["col1", "col2", "col3"]
        y_labels = ["Value 1", "Value 2", "Value 3"]
        titles = ["Plot 1", "Plot 2", "Plot 3"]

        # Create and save the violin plots
        violinplot_multiple_cols(dataframe=df, cols=cols, y_labels=y_labels, titles=titles, out_path="output/violin_plots.png")

    Further Details
    ---------------
    - Edge Cases: The function handles missing columns by raising a KeyError and suggesting similar column names.
    - Customizability: Users can customize the y-axis labels, plot titles, and output path for saving the plots.
    - Integration: The function can be used as part of larger data analysis workflows, integrating seamlessly with other plotting functions and data processing steps.

    This function is intended for researchers and developers who need to create detailed and customizable violin plots for visualizing data from a single DataFrame.

    Notes
    -----
    This function is part of the Plotting module and is designed to work in tandem with other plotting functions provided in the module.
    """
    # security
    for col in cols:
        check_for_col_in_df(col, dataframe)

    if not dims:
        dims = [None for _ in cols]

    if not titles:
        titles = ["" for _ in cols]

    def set_violinstyle(axes_subplot_parts) -> None:
        '''
        '''
        for pc in axes_subplot_parts["bodies"]:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        axes_subplot_parts["cmins"].set_edgecolor("black")
        axes_subplot_parts["cmaxes"].set_edgecolor("black")

    fig, ax_list = plt.subplots(1, len(cols), figsize=(3*len(cols), 5))
    fig.subplots_adjust(wspace=1, hspace=0.8)

    for ax, col, label, dim, title in zip(ax_list, cols, y_labels, dims, titles):
        ax.set_ylabel(label, size=13)
        ax.set_xticks([])
        data = dataframe[col].to_list()
        parts = ax.violinplot(dataframe[col].to_list(), widths=0.5)
        if dim:
            ax.set_ylim(dim)
        if title:
            ax.set_title(title, size=15)
        set_violinstyle(parts)
        quartile1, median, quartile3 = np.percentile(data, [25, 50, 75])
        ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
        ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
        ax.vlines(1, np.min(data), np.max(data), color="k", linestyle="-", lw=2)

    plt.figtext(0.5, 0.05, f'n = {len(dataframe.index)}', ha='center', fontsize=12)

    if out_path:
        fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    if show_fig:
        fig.show()
    return None

def check_for_col_in_df(col: str, df: pd.DataFrame) -> None:
    '''Checks if :col: is in :df: and gives similar columns if not'''
    if col not in df.columns:
        similar_cols = [c for c in df.columns if col.split("_")[0] in c]
        raise KeyError(f"Column {col} not found in DataFrame. Did you mean any of these columns? {similar_cols}")

def violinplot_multiple_lists(lists: list, titles: list[str], y_labels: list[str], dims: list[tuple[float,float]] = None, out_path: str = None, show_fig: bool = True) -> None:
    """
    Create Multiple Violin Plots from Lists of Data
    ===============================================
    
    This function generates multiple violin plots from specified lists of numerical data. It allows for customization of axis labels, plot titles, and plot dimensions, and optionally saves the resulting plots to a specified file path.

    Parameters
    ----------
    lists (list[list[float]]): 
        A list of lists, where each inner list contains numerical data points to be visualized.
    titles (list[str]): 
        A list of titles for each plot.
    y_labels (list[str]): 
        A list of labels for the y-axes of the plots.
    dims (list[tuple[float, float]], optional): 
        A list of tuples specifying the y-axis limits for each plot. If not provided, the y-axis limits will be determined automatically.
    out_path (str, optional): 
        The file path where the plots should be saved. If not provided, the plots will be displayed without saving.
    show_fig (bool, optional): 
        Whether to display the plot. Defaults to True.

    Detailed Description
    --------------------
    The `violinplot_multiple_lists` function creates multiple violin plots to visualize the distributions of specified lists of numerical data. The function supports extensive customization options, including axis labels, plot titles, and plot dimensions. It ensures that each plot is clearly labeled and informative, providing a comprehensive view of the data distribution.

    The function:
    - Generates violin plots for each specified list of data.
    - Sets plot titles, y-axis labels, and y-axis limits according to the provided parameters.
    - Optionally saves the plots to the specified file path.

    Examples
    --------
    Here is an example of how to use the `violinplot_multiple_lists` function:

    .. code-block:: python

        from plotting import violinplot_multiple_lists

        # Define data
        data1 = [1, 2, 3, 4, 5]
        data2 = [2, 3, 4, 5, 6]
        data3 = [3, 4, 5, 6, 7]

        # Define parameters
        lists = [data1, data2, data3]
        titles = ["Plot 1", "Plot 2", "Plot 3"]
        y_labels = ["Value 1", "Value 2", "Value 3"]

        # Create and save the violin plots
        violinplot_multiple_lists(lists=lists, titles=titles, y_labels=y_labels, out_path="output/violin_plots.png")

    Further Details
    ---------------
    - Edge Cases: The function handles empty data lists by not attempting to plot and logging a message.
    - Customizability: Users can customize the y-axis labels, plot titles, and output path for saving the plots.
    - Integration: The function can be used as part of larger data analysis workflows, integrating seamlessly with other plotting functions and data processing steps.

    This function is intended for researchers and developers who need to create detailed and customizable violin plots for visualizing multiple lists of numerical data.

    Notes
    -----
    This function is part of the Plotting module and is designed to work in tandem with other plotting functions provided in the module.
    """
    if not dims: dims = [None for sublist in lists]
    def set_violinstyle(axes_subplot_parts) -> None:
        '''
        '''
        for pc in axes_subplot_parts["bodies"]:
            pc.set_facecolor('cornflowerblue')
            pc.set_edgecolor('black')
            pc.set_alpha(1)

        axes_subplot_parts["cmins"].set_edgecolor("black")
        axes_subplot_parts["cmaxes"].set_edgecolor("black")
        return None

    fig, ax_list = plt.subplots(1, len(lists), figsize=(3*len(lists), 5))
    fig.subplots_adjust(wspace=1, hspace=0.8)

    if len(lists) == 1: ax_list = [ax_list]

    for ax, sublist, title, label, dim in zip(ax_list, lists, titles, y_labels, dims):
        ax.set_title(title, size=15, y=1.05)
        ax.set_ylabel(label, size=13)
        ax.set_xticks([])
        parts = ax.violinplot(sublist, widths=0.5)
        if dim: ax.set_ylim(dim)
        set_violinstyle(parts)
        quartile1, median, quartile3 = np.percentile(sublist, [25, 50, 75])
        ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
        ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
        ax.vlines(1, np.min(sublist), np.max(sublist), color="k", linestyle="-", lw=2)

    if out_path:
        fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    if show_fig:
        fig.show()
    return None

def scatterplot(dataframe:pd.DataFrame, x_column:str, y_column: str, color_column: str = None, size_column: str = None, labels: list[str] = None, title: str =None, show_corr: bool = False, out_path: str = None, show_fig: bool = False):
    """
    Create a Scatter Plot from a DataFrame
    ======================================
    
    This function generates a scatter plot from specified columns of a Pandas DataFrame. It allows for optional customization of point colors and sizes, as well as plot labels and title. The resulting plot can be saved to a specified file path or displayed.

    Parameters
    ----------
    dataframe (pd.DataFrame): 
        A Pandas DataFrame containing the data to be visualized.
    x_column (str): 
        The column name to be used for the x-axis data.
    y_column (str): 
        The column name to be used for the y-axis data.
    color_column (str, optional): 
        The column name to be used for point colors. Defaults to None.
    size_column (str, optional): 
        The column name to be used for point sizes. Defaults to None.
    labels (list[str], optional): 
        A list of labels for the axes and optional color and size legends. Defaults to None.
    title (str, optional): 
        The title of the plot. Defaults to None.
    out_path (str, optional): 
        The file path where the plot should be saved. If not provided, the plot will be displayed without saving.
    show_fig (bool, optional): 
        Whether to display the plot. Defaults to False.

    Detailed Description
    --------------------
    The `scatterplot` function creates a scatter plot to visualize relationships between two variables in a Pandas DataFrame. The function supports optional customization of point colors and sizes, allowing for additional dimensions of data to be represented visually. Custom axis labels, plot title, and legends can be provided to enhance the clarity and informativeness of the plot.

    The function:
    - Validates the presence of specified columns in the DataFrame.
    - Generates a scatter plot using the specified x and y columns.
    - Optionally colors points based on a specified column.
    - Optionally sizes points based on a specified column.
    - Sets plot title and axis labels according to the provided parameters.
    - Optionally saves the plot to the specified file path.

    Examples
    --------
    Here is an example of how to use the `scatterplot` function:

    .. code-block:: python

        from plotting import scatterplot
        import pandas as pd

        # Create a sample DataFrame
        df = pd.DataFrame({
            'x_data': [1, 2, 3, 4, 5],
            'y_data': [5, 4, 3, 2, 1],
            'color_data': [10, 20, 30, 40, 50],
            'size_data': [100, 200, 300, 400, 500]
        })

        # Define parameters
        x_column = 'x_data'
        y_column = 'y_data'
        color_column = 'color_data'
        size_column = 'size_data'
        labels = ["X Axis", "Y Axis", "Color Legend", "Size Legend"]
        title = "Sample Scatter Plot"

        # Create and save the scatter plot
        scatterplot(dataframe=df, x_column=x_column, y_column=y_column, color_column=color_column, size_column=size_column, labels=labels, title=title, out_path="output/scatter_plot.png", show_fig=True)

    Further Details
    ---------------
    - Edge Cases: The function handles missing columns by raising a KeyError and suggesting similar column names.
    - Customizability: Users can customize the axis labels, plot title, color and size columns, and output path for saving the plot.
    - Integration: The function can be used as part of larger data analysis workflows, integrating seamlessly with other plotting functions and data processing steps.

    This function is intended for researchers and developers who need to create detailed and customizable scatter plots for visualizing relationships between variables in a DataFrame.

    Notes
    -----
    This function is part of the Plotting module and is designed to work in tandem with other plotting functions provided in the module.
    """
    def evenly_spaced_values(value1, value2, num_values):
        # Calculate the step size
        step = (value2 - value1) / (num_values - 1)
        # Generate the evenly spaced values
        spaced_values = [value1 + i * step for i in range(num_values)]
        return spaced_values

    # define axes label names
    expected_labels = 2
    if color_column:
        expected_labels += 1
    if size_column:
        expected_labels += 1

    if not labels:
        labels = {"x": x_column, "y": y_column, "c": color_column, "s": size_column}
    else:
        num_labels = len(labels)
        if num_labels != expected_labels:
            raise ValueError("Number of labels must be the same as number of columns!")
        labels = {"x": labels[0], "y": labels[1], "c": labels[2] if color_column else None, "s": labels[3] if color_column and size_column else labels[2] if color_column else None}

    x_data = dataframe[x_column]
    y_data = dataframe[y_column]

    if color_column:
        color_values = dataframe[color_column]
        cmap = 'viridis'  # Choose a colormap for the color gradient
    else:
        color_values = None
        cmap = None


    # Create a figure with two subplots
    if size_column:
        size_values = dataframe[size_column]
        max_size = np.max(size_values)  # Get the maximum size for normalization
        sizes = 100 * (size_values / max_size)  # Scale sizes relative to max_size
        size_label = size_column
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), gridspec_kw={'width_ratios': [3, 0.5]})

        # Scatter plot on the first subplot
        scatter = ax1.scatter(x_data, y_data, c=color_values, cmap=cmap, s=sizes, alpha=0.5)
        ax1.set_xlabel(x_column)
        ax1.set_ylabel(y_column)

        if color_column:
            # Add color bar legend
            cbar = plt.colorbar(scatter, ax=ax1, label=labels["c"])

        # Size legend on the second subplot
        for size in evenly_spaced_values(size_values.min(), size_values.max(), 5):
            size_label = size
            ax2.scatter([], [], s=100*(size/max_size), label=size, alpha=0.5)

        ax2.set_axis_off()  # Hide axes for the size legend subplot
        ax2.legend(title=labels["s"], loc='center')

        # Adjust spacing between subplots
        plt.subplots_adjust(wspace=0)

    else:
        plt.figure(figsize=(8, 6))

        scatter = plt.scatter(x_data, y_data, c=color_values, cmap=cmap, alpha=0.5)

        # Add labels and title
        plt.xlabel(labels["x"])
        plt.ylabel(labels["y"])

        # Add color bar
        if color_column:
            plt.colorbar(scatter, label=labels["c"])

    if title:
        plt.suptitle(title, size=20)

    if show_corr:
        # Calculate Pearson correlation coefficient
        corr_coef = np.corrcoef(x_data, y_data)[0, 1]
        plt.text(0.05, 0.95, f'Correlation: {corr_coef:.2f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')

        # Calculate and plot line of best fit
        slope, intercept = np.polyfit(x_data, y_data, 1)
        best_fit_line = slope * x_data + intercept
        plt.plot(x_data, best_fit_line, color='red', linestyle='--', linewidth=2, label=f'y={slope:.2f}x+{intercept:.2f}')
        plt.legend()

    # Save the plot as a PNG file if out_path is provided
    if out_path:
        plt.savefig(out_path, dpi=300)
        logging.info(f"Plot saved as {out_path}")

    # Show the plot
    if show_fig:
        plt.show()

def parse_cols_for_plotting(plot_arg: str, subst:str=None) -> list[str]:
    """
    Parse Columns for Plotting
    ==========================
    
    This function processes the input argument to determine which columns should be used for plotting. It supports different input types and returns a list of column names.

    Parameters
    ----------
    plot_arg (str): 
        The argument indicating which columns to parse. It can be a string, list of strings, or a boolean.
    subst (str, optional): 
        A substitute string to use if `plot_arg` is a boolean set to True. Defaults to None.

    Detailed Description
    --------------------
    The `parse_cols_for_plotting` function is designed to handle various input formats for specifying columns to be used in plotting functions. It ensures that the returned value is always a list of strings, which can then be used to access the appropriate columns in a DataFrame.

    The function:
    - Converts a single string argument into a list containing that string.
    - Validates and returns a list of strings if the input is already a list.
    - Substitutes the provided string if `plot_arg` is set to True.
    - Raises a TypeError if the input argument type is unsupported.

    Examples
    --------
    Here is an example of how to use the `parse_cols_for_plotting` function:

    .. code-block:: python

        from plotting import parse_cols_for_plotting

        # Define input arguments
        plot_arg_str = "column1"
        plot_arg_list = ["column1", "column2"]
        plot_arg_bool = True
        subst = "default_column"

        # Parse columns for plotting
        cols_from_str = parse_cols_for_plotting(plot_arg=plot_arg_str)
        cols_from_list = parse_cols_for_plotting(plot_arg=plot_arg_list)
        cols_from_bool = parse_cols_for_plotting(plot_arg=plot_arg_bool, subst=subst)

        print(cols_from_str)  # Output: ['column1']
        print(cols_from_list)  # Output: ['column1', 'column2']
        print(cols_from_bool)  # Output: ['default_column']

    Further Details
    ---------------
    - Edge Cases: The function handles different input types gracefully, ensuring that a list of strings is always returned.
    - Customizability: Users can provide a substitute string to use if the input argument is a boolean set to True.
    - Integration: The function can be used as part of larger data analysis workflows, integrating seamlessly with other functions that require column names for plotting.

    This function is intended for researchers and developers who need to dynamically determine columns for plotting based on various input formats.

    Notes
    -----
    This function is part of the Plotting module and is designed to work in tandem with other plotting functions provided in the module.
    """
    if isinstance(plot_arg, str):
        return [plot_arg]
    elif isinstance(plot_arg, list):
        return plot_arg
    elif plot_arg == True:
        return [subst]
    else:
        raise TypeError("Unsupported argument type for parse_cols_for_plotting(): {type(plot_arg)}. Only list, str or bool allowed.")

def sequence_logo(dataframe: pd.DataFrame, input_col:str, out_path:str, refseq:str=None, title:str=None, resnums:list=None, units:str="probability"):
    """
    Generate a Sequence Logo
    ========================
    
    This function generates a sequence logo from a column of sequences in a Pandas DataFrame. It allows for customization of the reference sequence, plot title, residue numbers, and units used in the logo. The resulting plot is saved to a specified file path.

    Parameters
    ----------
    dataframe (pd.DataFrame): 
        A Pandas DataFrame containing the sequences to be visualized.
    input_col (str): 
        The column name containing the sequences or paths to fasta files.
    out_path (str): 
        The file path where the sequence logo should be saved.
    refseq (str, optional): 
        A reference sequence or a path to a fasta file containing the reference sequence. Defaults to None.
    title (str, optional): 
        The title of the sequence logo. Defaults to None.
    resnums (list, optional): 
        A list of integers specifying residue positions to include in the logo. Defaults to None.
    units (str, optional): 
        The units used in the sequence logo, either "probability" or "bits". Defaults to "probability".

    Detailed Description
    --------------------
    The `sequence_logo` function creates a sequence logo to visualize the conservation and variability of sequences. The function supports extensive customization options, including specifying a reference sequence, selecting specific residue positions, and choosing the units for the logo. It ensures that the resulting logo is clearly labeled and informative, providing insights into sequence conservation.

    The function:
    - Validates the presence of the specified input column in the DataFrame.
    - Prepares input sequences by handling both direct sequences and paths to fasta files.
    - Generates a sequence logo using the weblogo library.
    - Customizes the logo with the specified title, reference sequence, residue positions, and units.
    - Saves the sequence logo to the specified file path.

    Examples
    --------
    Here is an example of how to use the `sequence_logo` function:

    .. code-block:: python

        from plotting import sequence_logo
        import pandas as pd

        # Create a sample DataFrame
        df = pd.DataFrame({
            'sequences': ['ATGCGT', 'ATGCGC', 'ATGCGG']
        })

        # Define parameters
        input_col = 'sequences'
        out_path = 'output/sequence_logo.eps'
        refseq = 'ATGCGT'
        title = 'Sample Sequence Logo'
        resnums = [1, 2, 3, 4, 5, 6]
        units = 'bits'

        # Generate and save the sequence logo
        sequence_logo(dataframe=df, input_col=input_col, out_path=out_path, refseq=refseq, title=title, resnums=resnums, units=units)

    Further Details
    ---------------
    - Edge Cases: The function handles different input formats for sequences, including direct sequences and paths to fasta files. It also checks for consistent residue lengths.
    - Customizability: Users can customize the reference sequence, residue positions, plot title, and units for the sequence logo.
    - Integration: The function can be used as part of larger data analysis workflows, integrating seamlessly with other functions for sequence analysis and visualization.

    This function is intended for researchers and developers who need to create detailed and customizable sequence logos for visualizing sequence conservation and variability.

    Notes
    -----
    This function is part of the Plotting module and is designed to work in tandem with other plotting functions provided in the module.
    """
    import weblogo

    class RefSeqColor(weblogo.ColorScheme):
        """
        Color the given reference sequence in its own color, so you can easily see 
        which positions match that sequence and which don't.
        """
        def __init__(self, ref_seq, color, description=None):
            self.ref_seq = ref_seq
            self.color = weblogo.Color.from_string(color)
            self.description = description

        def symbol_color(self, seq_index, symbol, rank):
            if symbol == self.ref_seq[seq_index]:
                return self.color

    def extract_extension(file_path):
        '''extract file extension'''
        _, ext = os.path.splitext(file_path)
        return ext

    def prepare_input(tmp_fasta:str, dataframe:pd.DataFrame, input_col:str):
        '''check format of input sequences and write all sequences to a single fasta'''
        ext = list(set(dataframe[input_col].apply(extract_extension).to_list()))
        seqs = dataframe[input_col].to_list()
        if len(ext) > 1:
            raise RuntimeError("Input column must contain either sequences or paths to fasta files that end with .fa or .fasta!")
        if ext == [""]:
            with open(tmp_fasta, 'a', encoding="UTF-8") as t:
                for i, seq in enumerate(seqs):
                    t.write(f">{i}\n{seq}\n")
        elif ext[0] in [".fasta", ".fa"]:
            for fa in seqs:
                with open(fa, 'r', encoding="UTF-8") as f:
                    content = f.read()
                with open(tmp_fasta, 'a', encoding="UTF-8") as t:
                    t.write(content)

    def prepare_reference_seq(refseq:str):
        '''check format of reference sequence and extract sequence from fasta if necessary'''
        if refseq.endswith('.fa') or refseq.endswith('.fasta'):
            seq = ""
            seq_dict = import_fasta(refseq)
            if len(seq_dict) > 1:
                raise RuntimeError("Reference sequence fasta must contain only a single entry!")
            _, seq = next(iter(seq_dict.items()))
            return seq
        return refseq

    def replace_logo_numbers(eps_str, resid, sequence=None):
        '''
        Changes the weblogo output to have correct x-labels using correct residuenumbers.
        '''
        new_eps_str_lines = []
        pos_count = 0
        for line in eps_str.split('\n'):
            if line.endswith('StartStack') and not line.startswith('%'):
                if sequence is None:
                    line = line.replace(str(pos_count + 1), '%d' % (resid[pos_count]))
                else:
                    line = line.replace(str(pos_count + 1), '%s%d' % (sequence[pos_count],resid[pos_count]))
                pos_count=pos_count+1
            new_eps_str_lines.append(line)
        return '\n'.join(new_eps_str_lines)

    def modify_input_fasta_based_on_residues(in_fasta:str, out_fasta:str, resnums:list):
        '''create a new fasta file containing only the residues according to <resnums>'''
        seq_dict = import_fasta(in_fasta)
        for id in seq_dict:
            seq_dict[id] = ''.join([seq_dict[id][i-1] for i in resnums])
        write_fasta(seq_dict=seq_dict, fasta=out_fasta)


    check_for_col_in_df(col=input_col, df=dataframe)
    tmp_fasta = "tmp_seqlogo.fasta"
    # check if input are sequences or fasta files and create a temporary fasta file
    prepare_input(tmp_fasta=tmp_fasta, dataframe=dataframe, input_col=input_col)

    # check if only certain residues should be used for the creation of the sequence logo
    if resnums: modify_input_fasta_based_on_residues(in_fasta=tmp_fasta, out_fasta=tmp_fasta, resnums=resnums)

    # set sequence logo parameters
    protein_alphabet=weblogo.Alphabet('ACDEFGHIKLMNPQRSTVUWY', zip('acdefghiklmnpqrstvuwy','ACDEFGHIKLMNPQRSTVUWY'))
    baserules = [weblogo.SymbolColor("GSTYC", "green", "polar"),
                weblogo.SymbolColor("NQ", "purple", "neutral"),
                weblogo.SymbolColor("KRH", "blue", "basic"),
                weblogo.SymbolColor("DE", "red", "acidic"),
                weblogo.SymbolColor("PAWFLIMV", "black", "hydrophobic")
            ]
    colorscheme = weblogo.ColorScheme(baserules, alphabet=protein_alphabet)

    with open(tmp_fasta, 'r', encoding="UTF-8") as f:
        seqs = weblogo.read_seq_data(f, alphabet=protein_alphabet)
    os.remove(tmp_fasta)

    # check if reference sequence is provided and extract sequence
    if refseq:
        refseq = prepare_reference_seq(refseq=refseq)
        colorscheme = weblogo.ColorScheme([RefSeqColor(refseq, "orange", "refseq")] + baserules, alphabet=protein_alphabet)

    data = weblogo.LogoData.from_seqs(seqs)
    options = weblogo.LogoOptions()
    options.logo_title = title
    options.unit_name=units
    options.show_fineprint = False
    options.color_scheme = colorscheme
    options.xaxis_tic_interval = 1
    options.number_interval = 1
    options.number_fontsize = 3
    logoformat = weblogo.LogoFormat(data, options)
    eps_binary = weblogo.eps_formatter(data, logoformat)
    eps_str = eps_binary.decode()
    # reindex sequence based on resnums
    if resnums is not None:
        if refseq is None:
            eps_str = replace_logo_numbers(eps_str, resnums)
        else:
            eps_str = replace_logo_numbers(eps_str, resnums, refseq)

    with open(out_path, 'w', encoding="UTF-8") as f:
        f.write(eps_str)


def write_fasta(seq_dict:dict, fasta:str):
    """
    Write Sequences to a Fasta File
    ===============================
    
    This function writes a dictionary of sequences to a fasta file. Each key-value pair in the dictionary represents a sequence identifier and its corresponding sequence.

    Parameters
    ----------
    seq_dict (dict): 
        A dictionary where keys are sequence identifiers and values are sequences.
    fasta (str): 
        The file path where the fasta file should be saved.

    Detailed Description
    --------------------
    The `write_fasta` function is designed to facilitate the creation of fasta files from a dictionary of sequences. This function iterates over the dictionary and writes each sequence to the specified file in the standard fasta format.

    The function:
    - Opens the specified file for writing.
    - Iterates over the dictionary, writing each sequence identifier and sequence to the file.
    - Ensures that the file is saved in the correct fasta format.

    Examples
    --------
    Here is an example of how to use the `write_fasta` function:

    .. code-block:: python

        from plotting import write_fasta

        # Define a sequence dictionary
        seq_dict = {
            'seq1': 'ATGCGT',
            'seq2': 'ATGCGC',
            'seq3': 'ATGCGG'
        }

        # Define the output path
        fasta = 'output/sequences.fasta'

        # Write the sequences to the fasta file
        write_fasta(seq_dict=seq_dict, fasta=fasta)

    Further Details
    ---------------
    - Edge Cases: The function handles dictionaries with varying sequence lengths, ensuring each sequence is written correctly.
    - Customizability: Users can specify any valid file path for the output fasta file.
    - Integration: The function can be used as part of larger workflows involving sequence analysis and data export.

    This function is intended for researchers and developers who need to export sequences to a fasta file for further analysis or sharing.
    """
    with open(fasta, 'w', encoding="UTF-8") as f:
        for id_ in seq_dict:
            f.write(f">{id_}\n{seq_dict[id_]}\n")

def import_fasta(fasta:str):
    """
    Import Sequences from a Fasta File
    ==================================
    
    This function imports sequences from a fasta file and returns them as a dictionary. Each key-value pair in the dictionary represents a sequence identifier and its corresponding sequence.

    Parameters
    ----------
    fasta (str): 
        The file path of the fasta file to be imported.

    Detailed Description
    --------------------
    The `import_fasta` function reads a fasta file and parses the sequences into a dictionary format. This function is useful for loading sequences into a program for further analysis or manipulation.

    The function:
    - Opens the specified fasta file for reading.
    - Parses the file content, extracting sequence identifiers and sequences.
    - Returns a dictionary where keys are sequence identifiers and values are sequences.

    Examples
    --------
    Here is an example of how to use the `import_fasta` function:

    .. code-block:: python

        from plotting import import_fasta

        # Define the input path
        fasta = 'input/sequences.fasta'

        # Import the sequences from the fasta file
        seq_dict = import_fasta(fasta=fasta)

        # Print the imported sequences
        print(seq_dict)

    Further Details
    ---------------
    - Edge Cases: The function handles fasta files with varying sequence lengths and multiple sequences, ensuring all sequences are parsed correctly.
    - Customizability: Users can specify any valid file path for the input fasta file.
    - Integration: The function can be used as part of larger workflows involving sequence analysis and data import.

    This function is intended for researchers and developers who need to import sequences from a fasta file for analysis or manipulation.
    """
    with open(fasta, 'r', encoding="UTF-8") as f:
        fastas = f.read()
    # split along > (separator)
    raw_fasta_list = [x.strip().split("\n") for x in fastas.split(">") if x]
    # parse into dictionary {description: sequence}
    fasta_dict = {x[0]: "".join(x[1:]) for x in raw_fasta_list if len(x) > 1}
    return fasta_dict
