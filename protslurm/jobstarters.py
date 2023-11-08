'''
Jobstarters contains various jobstarter objects for SLURM or other systems
that are used by runners to start jobs.
'''
import time
import subprocess
import itertools

class JobStarter:
    '''JobStarter class is a class that defines how jobstarters have to look.'''
    def start(self, cmds:list, options:str, jobname:str, wait:bool) -> None:
        '''Method to start jobs'''
        raise NotImplementedError("Jobstarter 'start' function was not overwritten!")

    def wait_for_job(self, jobname:str, interval:float) -> None:
        '''Method for waiting for started jobs'''
        raise NotImplementedError("Jobstarter 'wait_for_job' function was not overwritten!")

class SbatchArrayJobstarter(JobStarter):
    '''Jobstarter that starts Job arrays on slurm clusters.'''
    def __init__(self, max_array_size:int=100, remove_cmdfile:bool=True):
        self.max_array_size = max_array_size
        self.remove_cmdfile = remove_cmdfile

        # static attribute, can be changed depending on slurm settings:
        self.slurm_max_arrayjobs = 1000

    def start(self, cmds:list, options:str, jobname:str, wait:bool=True, cmdfile_dir:str="./") -> None:
        '''
        Writes [cmds] into a cmd_file that contains each cmd in a separate line.
        Then starts an sbatch job running down the cmd-file.
        '''
        # check if cmds is smaller than 1000. If yes, split cmds and start split array!
        if len(cmds) > self.slurm_max_arrayjobs:
            print(f"The commands-list you supplied is longer than self.slurm_max_arrayjobs. Your job will be subdivided into multiple arrays.")
            for sublist in split_list(cmds, self.slurm_max_arrayjobs):
                self.start(cmds=sublist, options=options, jobname=jobname, wait=wait, cmdfile_dir=cmdfile_dir)
            return None

        # write cmd-file
        jobname = add_timestamp(jobname)
        with open((cmdfile := f"{cmdfile_dir}/{jobname}_cmds"), 'w', encoding="UTF-8") as f:
            f.write("\n".join(cmds))

        # write sbatch command and run
        sbatch_cmd = f'sbatch -a 1-{str(len(cmds))}%{str(self.max_array_size)} -J {jobname} -vvv {" ".join(options)} --wrap "eval {chr(92)}`sed -n {chr(92)}${{SLURM_ARRAY_TASK_ID}}p {cmdfile}{chr(92)}`"'
        subprocess.run(sbatch_cmd, shell=True, stdout=True, stderr=True, check=True)

        # wait for job and clean up
        if wait: self.wait_for_job(jobname)
        if self.remove_cmdfile: subprocess.run(f"rm {cmdfile}", shell=True, stdout=True, stderr=True, check=True)
        return None

    def wait_for_job(self, jobname:str, interval:float=5) -> None:
        '''
        Waits for slurm jobs to be finished.
        '''
        # Check if job is running by capturing the length of the output of squeue command that only returns jobs with <jobname>:
        while len(subprocess.run(f'squeue -n {jobname} -o "%A"', shell=True, capture_output=True, text=True, check=True).stdout.strip().split("\n")) > 1:
            time.sleep(interval)
        print(f"Job {jobname} completed.\n")
        time.sleep(10)
        return None

def add_timestamp(x: str) -> str:
    '''
    Adds a unique (in most cases) timestamp to a string using the "time" library.
    Returns string with timestamp added to it.
    '''
    return "_".join([x, f"{str(time.time()).rsplit('.', maxsplit=1)[-1]}"])

def split_list(input_list: list, element_length: int) -> list:
    '''Splits 'input_list' into nested list of sublists with maximum length of 'element_length' '''
    result = []
    iterator = iter(input_list)
    while True:
        sublist = list(itertools.islice(iterator, element_length))
        if not sublist:
            break
        result.append(sublist)
    return result
