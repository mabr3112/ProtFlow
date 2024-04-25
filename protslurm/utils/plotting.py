##################################### PLOTTING ###########################################################
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import rc
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
if __file__.startswith("/home/mabr3112"):
    matplotlib.use('Agg')
else:
    print("Using Matplotlib without 'Agg' backend.")

import numpy as np

class PlottingTrajectory():
    def __init__(self, y_label: str, location: str, title:str="Refinement Trajectory", dims=None):
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
        if not type(data_list) == list:
            data_list = list(data_list)
        self.data.append((label, data_list))
        return None
    
    def violin_plot(self, out_path:str=None):
        '''AAA'''
        out_path = out_path or self.location
        if not self.data: return print(f"Nothing can be plotted, no data added yet.")
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
        fig, ax = plt.subplots(1, 1, figsize=(3+0.8*(len(self.data)), 5))

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

        handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, [x[0] for x in self.data])]
        fig.legend(handles=handles, loc='right', bbox_to_anchor=(0.45+(1-len(self.data)*0.05),0.5),
                  fancybox=True, shadow=True, ncol=1, fontsize=13)

        if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
        else: fig.show()
        return None
        
    def add_and_plot(self, data_list, label):
        ''''''
        self.add(data_list, label)
        self.violin_plot()
        return None
        
def singular_violinplot(data: list, y_label: str, title: str, out_path:str=None,) -> None:
    '''AAA'''
    fig, ax = plt.subplots(figsize=(2,5))

    parts = ax.violinplot(distances, widths=0.5)
    ax.set_title(title, fontsize=18)
    ax.set_ylabel(y_label, size=13) # "\u00C5" is Unicode for Angstrom
    ax.set_xticks([])
    
    quartile1, median, quartile3 = np.percentile(distances, [25, 50, 75]) #axis=1 if multiple violinplots.
    
    for pc in parts['bodies']:
        pc.set_facecolor('cornflowerblue')
        pc.set_edgecolor('black')
        pc.set_alpha(1)
    
    parts["cmins"].set_edgecolor("black")
    parts["cmaxes"].set_edgecolor("black")
    print([x for x in parts["bodies"]])
        
    ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
    ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
    ax.vlines(1, np.min(distances), np.max(distances), color="k", linestyle="-", lw=2)
    
    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_cols_dfs(dfs, df_names, cols, titles, y_labels, dims=None, out_path=None, colormap="tab20") -> None:
    ''''''
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

        handles = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, df_names)]
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.1),
                  fancybox=True, shadow=True, ncol=5, fontsize=13)
        
    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_cols(df, cols, titles, y_labels, dims=None, out_path=None) -> None:
    '''AAA'''
    if not dims: dims = [None for col in cols]
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

    fig, ax_list = plt.subplots(1, len(cols), figsize=(3*len(cols), 5))
    fig.subplots_adjust(wspace=1, hspace=0.8)

    for ax, col, title, label, dim in zip(ax_list, cols, titles, y_labels, dims):
        ax.set_title(title, size=15, y=1.05)
        ax.set_ylabel(label, size=13)
        ax.set_xticks([])
        data = df[col].to_list()
        parts = ax.violinplot(df[col].to_list(), widths=0.5)
        if dim: ax.set_ylim(dim)
        set_violinstyle(parts)
        quartile1, median, quartile3 = np.percentile(data, [25, 50, 75])
        ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
        ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
        ax.vlines(1, np.min(data), np.max(data), color="k", linestyle="-", lw=2)

    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_lists(lists: list, titles: list[str], y_labels: list[str], dims=None, out_path=None) -> None:
    '''AAA'''
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

    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def parse_cols_for_plotting(plot_arg: str, subst:str=None) -> list[str]:
    '''AAA'''
    if type(plot_arg) == str: return [plot_arg]
    elif type(plot_arg) == list: return plot_arg
    elif plot_arg == True: return [subst]
    else: raise TypeError("Unsupported argument type for parse_cols_for_plotting(): {type(plot_arg)}. Only list, str or bool allowed.")
