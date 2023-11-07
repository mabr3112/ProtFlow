'''
Jobstarters contains various jobstarter objects for SLURM or other systems
that are used by runners to start jobs.
'''

class JobStarter:
    '''JobStarter class is a class that defines how jobstarters have to look.'''
    def __init__(self):
        print("WEEE")

    def start(self) -> None:
        '''Method to start jobs'''
        return None
