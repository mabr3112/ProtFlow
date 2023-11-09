'''Module Containing python runners'''
import pandas as pd
from protslurm.jobstarters import JobStarter

class RunnerOutput:
    '''RunnerOutput class handles how protein data is passed between Runners and Poses classes.'''
    def __init__(self, data:pd.DataFrame=None, index_layers:int=0):
        self.set_df(data=data)
        self.index_layers = index_layers # how many index layers were added to the description of the poses as a result of the Runner?

    def set_df(self, data: pd.DataFrame) -> None:
        '''Method to set data to RunnerOutput retrospectively (for whatever reason)'''
        self.df = data
        if data is None: return

        # check formatting:
        mandatory_cols = ["description", "location"]
        if mandatory_cols not in data.columns: raise ValueError(f"Input Data to RunnerOutput class MUST contain columns 'description' and 'location'.\nDescription should carry the name of the poses, while 'location' should contain the path (+ filename and suffix).")

class Runner:
    '''Abstract Runner baseclass handling interface between Runners and Poses.'''
    def __str__(self):
        raise NotImplementedError(f"Your Runner needs a name! Set in your Runner class: 'def __str__(self): return \"runner_name\"'")

    def run(self, poses:list, jobstarter:JobStarter, output_dir:str, options:str=None, pose_options:str=None) -> RunnerOutput:
        '''method that interacts with Poses to run jobs and send Poses the scores.'''
        raise NotImplementedError(f"Runner Method 'run' was not overwritten yet!")

def parse_generic_options(options: str, pose_options: str, sep="--") -> tuple[dict,list]:
    '''Parses options and pose_options together into options (dict {arg: val, ...}) and flags (list: [val, ...])'''
    def tmp_parse_options_flags(options_str: str, sep:str="--") -> tuple[dict, list]:
        '''parses split options '''
        if not options_str: return {}, []
        # split along separator
        firstsplit = [x.strip() for x in options.split(sep) if x]

        # parse into options and flags:
        opts = dict()
        flags = list()
        for item in firstsplit:
            if len((x := item.split())) > 1:
                opts[x[0]] = " ".join(x[1:])
            else:
                flags.append(x[0])

        return opts, flags

    # parse into options and flags:
    opts, flags = tmp_parse_options_flags(options, sep=sep)
    pose_opts, pose_flags = tmp_parse_options_flags(pose_options, sep=sep)

    # merge options and pose_options (pose_opts overwrite opts), same for flags
    opts.update(pose_opts)
    flags = list(set(flags) | set(pose_flags))

    return opts, flags
