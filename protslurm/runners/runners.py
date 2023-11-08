'''Module Containing python runners'''
import pandas as pd
from protslurm.jobstarters import JobStarter

class RunnerOutput:
    '''RunnerOutput class handles how protein data is passed between Runners and Poses classes.'''
    def __init__(self, data:pd.DataFrame=None, index_layers:int=0):
        self.set_data(data=data)
        self.index_layers = index_layers # how many index layers were added to the description of the poses as a result of the Runner?

    def set_data(self, data: pd.DataFrame) -> None:
        '''Method to set data to RunnerOutput retrospectively (for whatever reason)'''
        self.data = data
        if data is None: return

        # check formatting:
        mandatory_cols = ["description", "location"]
        if mandatory_cols not in data.columns: raise ValueError(f"Input Data to RunnerOutput class MUST contain columns 'description' and 'location'.\nDescription should carry the name of the poses, while 'location' should contain the path (+ filename and suffix).")

class Runner:
    '''Abstract Runner baseclass handling interface between Runners and Poses.'''
    def __init__(self):
        runner=""
        print(runner)

    def run(self, prefix:str, jobstarter:JobStarter, output_dir:str, options:str=None, pose_options:str=None) -> RunnerOutput:
        '''method that interacts with Poses to run jobs and send Poses the scores.'''
        raise NotImplementedError(f"Runner Method 'run' was not overwritten yet!")
    
    def check_output(self) -> None:
        '''Method to check if runner has ran already. If so it should just read in the output and skip running.'''
        raise NotImplementedError(f"Runner Method check_output was not overwritten yet!")
