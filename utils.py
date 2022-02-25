import os
import datetime

# Log
class Logger:
    def __init__(self, experiment_time, filename='log'):
        if not os.path.exists('./results'):
            os.mkdir('./results')
        self.logdir = os.path.join('./results', experiment_time)
        os.makedirs(self.logdir)
        self.logfile = os.path.join(self.logdir, filename)
        self.print_and_write("Experiment started at " + experiment_time)

    def print_and_write(self, log):
        print(log)
        with open(self.logfile, 'a') as f:
            f.write(log + '\n')

    def write(self, log):
        with open(self.logfile, 'a') as f:
            f.write(log + '\n')