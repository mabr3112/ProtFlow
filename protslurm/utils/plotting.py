'''
Module for creating standardized plots (violins mostyl)
'''
# dependencies
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

def singular_violinplot(data: list, y_label: str, title: str, out_path: str = None,) -> None:
    '''AAA'''
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
    print([x for x in parts["bodies"]])

    ax.scatter(1, median, marker='o', color="white", s=40, zorder=3)
    ax.vlines(1, quartile1, quartile3, color="k", linestyle="-", lw=10)
    ax.vlines(1, np.min(data), np.max(data), color="k", linestyle="-", lw=2)

    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_cols_dfs(dfs, df_names, cols, titles, y_labels, dims=None, out_path=None, colormap="tab20") -> None:
    '''Creates a violinplot of multiple columns from multiple Pandas DataFrames. Ideal for comparing stuff.'''
    # security
    for df in dfs:
        for col in cols:
            check_for_col_in_df(col, df)

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
        
    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def violinplot_multiple_cols(df, cols, titles, y_labels, dims=None, out_path=None) -> None:
    '''AAA'''
    # security
    for col in cols:
        check_for_col_in_df(col, df)

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

    plt.figtext(0.5, 0.05, f'n = {len(df.index)}', ha='center', fontsize=12)
    
    if out_path: fig.savefig(out_path, dpi=300, format="png", bbox_inches="tight")
    else: fig.show()
    return None

def check_for_col_in_df(col: str, df: pd.DataFrame) -> None:
    '''CHecks if column is in df and gives similar columns if not'''
    if col not in df.columns:
        similar_cols = [c for c in df.columns if col.split("_")[0] in c]
        raise KeyError(f"Column {col} not found in DataFrame. Did you mean any of these columns? {similar_cols}")

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


def scatterplot(dataframe, x_column, y_column, color_column=None, size_column=None, labels=None, save_path=None):
    """
    Create a scatter plot from a DataFrame using specified columns for x and y axes,
    and optionally use other columns for color gradient and dot size. Labels must be a list of strings, one element for each column used. Otherwise, column names will be used as labels.
    """

    def evenly_spaced_values(value1, value2, num_values):
        # Calculate the step size
        step = (value2 - value1) / (num_values - 1)
        # Generate the evenly spaced values
        spaced_values = [value1 + i * step for i in range(num_values)]
        return spaced_values
    
    # define axes label names
    expected_labels = 2        
    if color_column: expected_labels += 1
    if size_column: expected_labels += 1 

    if not labels:
        labels = {"x": x_column, "y": y_column, "c": color_column, "s": size_column}
    else:
        num_labels = len(labels)
        if not num_labels == expected_labels:
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

    # Save the plot as a PNG file if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved as {save_path}")

    # Show the plot
    plt.show()

def parse_cols_for_plotting(plot_arg: str, subst:str=None) -> list[str]:
    '''AAA'''
    if type(plot_arg) == str: return [plot_arg]
    elif type(plot_arg) == list: return plot_arg
    elif plot_arg == True: return [subst]
    else: raise TypeError("Unsupported argument type for parse_cols_for_plotting(): {type(plot_arg)}. Only list, str or bool allowed.")


        
def sequence_logo(dataframe:pd.DataFrame, input_col:str, save_path:str, refseq:str=None, title:str=None, resnums:list=None, units:str="probability"):
    '''
    Generates a WEBLOGO with adapted x-axes (having correct residue IDs as lables)

    Parameters
    ----------
    dataframe : pd.DataFrame
        input dataframe
    input_col : str
        name of input column. must contain either protein sequences or paths to fasta files that end with .fa or .fasta
    save_path : str
        Path where sequence logo should be stored
    refseq : str, optional
        If the native sequene is of interest, the sequence or the path to a fasta file can be given. This sequence must be as long as all other sequences!
        These residues will be colored in orange in the weblogo and displayed on the x-axis.
    title: str
        Write a title on your sequence logo
    resnums:
        Only create the sequence logo for certain residues. Must be a list of integers indicating the residue position (first residue is 1, not 0!)
    units: probability or bits
        which units should be used, bits are explained here: https://en.wikipedia.org/wiki/Sequence_logo#Logo_creation

    Returns 
    -------
    a weblogo

    '''
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
        elif ext[0] == [""]:
            with open(tmp_fasta, 'a') as t:
                for i, seq in enumerate(seqs):
                    t.write(f">{i}\n{seq}\n")
        elif ext[0] in [".fasta", ".fa"]:
            for fa in seqs:
                with open(fa, 'r') as f:
                    content = f.read()
                with open(tmp_fasta, 'a') as t:
                    t.write(content)
    
    def prepare_reference_seq(refseq:str):
        '''check format of reference sequence and extract sequence from fasta if necessary'''
        if refseq.endswith('.fa') or refseq.endswith('.fasta'):
            seq = ""
            seq_dict = import_fasta(refseq)
            if len([i for i in seq_dict]) > 1:
                raise RuntimeError("Reference sequence fasta must contain only a single entry!")
            id, seq = next(iter(seq_dict.items()))
            return seq
        else:
            return refseq
        
    def replace_logo_numbers(eps_str, resid, sequence=None):
        '''
        Changes the weblogo output to have correct x-labels using correct residuenumbers.
        '''
        new_eps_str_lines = []
        pos_count = 0
        for line in eps_str.split('\n'):
            if line.endswith('StartStack') and not line.startswith('%'):
                if sequence==None:
                    line = line.replace(str(pos_count + 1), '%d' % (resid[pos_count]))
                else:
                    line = line.replace(str(pos_count + 1), '%d%s' % (resid[pos_count], sequence[pos_count],))
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

    fasta_file = open(tmp_fasta, "r")
    seqs = weblogo.read_seq_data(fasta_file, alphabet=protein_alphabet)

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
        if refseq==None:
            eps_str = replace_logo_numbers(eps_str, resnums)      
        else:
            eps_str = replace_logo_numbers(eps_str, resnums, refseq)

    with open(save_path, 'w') as f:
        f.write(eps_str)


def write_fasta(seq_dict:dict, fasta:str):
    with open(fasta, 'w') as f:
        for id in seq_dict:
            f.write(f">{id}\n{seq_dict[id]}\n")

def import_fasta(fasta:str):
    with open(fasta, 'r') as f:
        fastas = f.read()
    # split along > (separator)
    raw_fasta_list = [x.strip().split("\n") for x in fastas.split(">") if x]
    # parse into dictionary {description: sequence}
    fasta_dict = {x[0]: "".join(x[1:]) for x in raw_fasta_list if len(x) > 1}
    return fasta_dict